// To benchmark without LLVM eliding compute loops, build with:
//   cargo build --release --features benchmark
// The `benchmark` feature wraps every output write in std::hint::black_box,
// preventing the compiler from proving the result is unused and deleting the
// loop.  Remove it for production builds — it suppresses autovectorisation.
//
// In Cargo.toml add:
//   [features]
//   benchmark = []

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use ndarray::{Axis, Zip};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::sync::OnceLock;

const EPSILON: f32 = 1e-10;

// Reciprocal LUT for sums of two u16 values.
// Range: 0..=131_070  (65_535 + 65_535)
// Size:  131_071 × 4 bytes = 524,284 bytes (~512 KB).
// This sits in L3 on most desktop CPUs; do not expect L2 residency.
static RECIPROCAL_LUT: OnceLock<Vec<f32>> = OnceLock::new();

fn init_hardware_optimization() {
    // ThreadPoolBuilder::build_global is itself idempotent (returns an error
    // on subsequent calls, which we intentionally discard).  OnceLock ensures
    // the LUT is computed exactly once regardless of call concurrency.

    // Use std::thread::available_parallelism instead of the num_cpus crate.
    // On a memory-bandwidth-bound workload like this one, logical CPUs
    // (including hyperthreads) often outperform physical-only counts because
    // the extra threads hide memory-fetch latency.  Benchmark both if tuning.
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let _ = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();

    // Iterator form: identical codegen to the explicit push loop, but idiomatic.
    RECIPROCAL_LUT.get_or_init(|| {
        (0u32..=131_070)
            .map(|i| 1.0_f32 / (i as f32 + EPSILON))
            .collect()
    });
}

// ---------------------------------------------------------------------------
// Shape validation
// ---------------------------------------------------------------------------

/// Validate that all provided shapes are identical, returning a PyValueError
/// that includes the actual shapes if they differ.
fn validate_shapes_inplace(a: &[usize], b: &[usize], out: &[usize]) -> PyResult<()> {
    if a != b || a != out {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: band_a={a:?}, band_b={b:?}, out_buffer={out:?}. \
             All arrays must have identical dimensions."
        )));
    }
    Ok(())
}

fn validate_shapes(a: &[usize], b: &[usize]) -> PyResult<()> {
    if a != b {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: band_a={a:?}, band_b={b:?}. \
             Input arrays must have identical dimensions."
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Core compute kernels (hand-written, maximally optimised)
// ---------------------------------------------------------------------------

/// Standard path: floating-point division with epsilon guard.
/// Prefer for tiles smaller than ~2048×2048 where the LUT causes cache pressure.
fn compute_logic(
    a_arr: ndarray::ArrayView2<u16>,
    b_arr: ndarray::ArrayView2<u16>,
    mut out: ndarray::ArrayViewMut2<f32>,
) {
    // Fast path: all three arrays are contiguous in memory — use flat slices
    // so Rayon can partition without ndarray chunk overhead.
    if let (Some(a_s), Some(b_s), Some(o_s)) =
        (a_arr.as_slice(), b_arr.as_slice(), out.as_slice_mut())
    {
        o_s.par_iter_mut()
            .zip(a_s.par_iter())
            .zip(b_s.par_iter())
            .for_each(|((out_val, &a), &b)| {
                let fa  = a as f32;
                let fb  = b as f32;
                let val = (fa - fb) / (fa + fb + EPSILON);
                #[cfg(feature = "benchmark")]
                { *out_val = std::hint::black_box(val); }
                #[cfg(not(feature = "benchmark"))]
                { *out_val = val; }
            });
    } else {
        // Slow path: non-contiguous layout (e.g. a strided slice from rasterio).
        // chunk_size is floored at 64 rows so that on machines where nrows <
        // thread_count we don't create chunks of 1 row and drown in scheduler
        // overhead.
        let chunk_size = (a_arr.nrows() / rayon::current_num_threads()).max(64);
        out.axis_chunks_iter_mut(Axis(0), chunk_size)
            .into_par_iter()
            .zip(a_arr.axis_chunks_iter(Axis(0), chunk_size))
            .zip(b_arr.axis_chunks_iter(Axis(0), chunk_size))
            .for_each(|((mut o_c, a_c), b_c)| {
                Zip::from(&mut o_c)
                    .and(&a_c)
                    .and(&b_c)
                    .for_each(|o, &a, &b| {
                        let fa  = a as f32;
                        let fb  = b as f32;
                        let val = (fa - fb) / (fa + fb + EPSILON);
                        #[cfg(feature = "benchmark")]
                        { *o = std::hint::black_box(val); }
                        #[cfg(not(feature = "benchmark"))]
                        { *o = val; }
                    });
            });
    }
}

/// LUT path: replaces division with a reciprocal multiply from a precomputed
/// table.  The LUT is ~512 KB and typically resides in L3 cache.
/// Prefer for tiles >= ~2048×2048 where sequential memory bandwidth dominates
/// and the LUT fetch cost is amortised across many elements.
fn compute_logic_lut(
    a_arr: ndarray::ArrayView2<u16>,
    b_arr: ndarray::ArrayView2<u16>,
    mut out: ndarray::ArrayViewMut2<f32>,
) {
    let lut = RECIPROCAL_LUT.get().expect(
        "RECIPROCAL_LUT accessed before init_hardware_optimization() was called."
    );

    if let (Some(a_s), Some(b_s), Some(o_s)) =
        (a_arr.as_slice(), b_arr.as_slice(), out.as_slice_mut())
    {
        o_s.par_iter_mut()
            .zip(a_s.par_iter())
            .zip(b_s.par_iter())
            .for_each(|((out_val, &a), &b)| {
                let sum  = a as usize + b as usize;
                let diff = a as f32 - b as f32;
                let val  = diff * lut[sum];
                #[cfg(feature = "benchmark")]
                { *out_val = std::hint::black_box(val); }
                #[cfg(not(feature = "benchmark"))]
                { *out_val = val; }
            });
    } else {
        let chunk_size = (a_arr.nrows() / rayon::current_num_threads()).max(64);
        out.axis_chunks_iter_mut(Axis(0), chunk_size)
            .into_par_iter()
            .zip(a_arr.axis_chunks_iter(Axis(0), chunk_size))
            .zip(b_arr.axis_chunks_iter(Axis(0), chunk_size))
            .for_each(|((mut o_c, a_c), b_c)| {
                Zip::from(&mut o_c)
                    .and(&a_c)
                    .and(&b_c)
                    .for_each(|o, &a, &b| {
                        let sum  = a as usize + b as usize;
                        let diff = a as f32 - b as f32;
                        let val  = diff * lut[sum];
                        #[cfg(feature = "benchmark")]
                        { *o = std::hint::black_box(val); }
                        #[cfg(not(feature = "benchmark"))]
                        { *o = val; }
                    });
            });
    }
}

// ---------------------------------------------------------------------------
// Stack-machine expression evaluator
// ---------------------------------------------------------------------------
//
// Architecture
// ~~~~~~~~~~~~
// The Python side parses a Spyndex formula string into a post-order bytecode
// list (e.g. "(N - R) / (N + R)" → [PushBand(0), PushBand(1), Sub,
// PushBand(0), PushBand(1), Add, Div]) and calls `compile_expr` once.
// The returned `CompiledExpr` object is then passed to `execute_expr_inplace`
// for each tile.  The tree is never walked again after compilation — the hot
// loop is a flat instruction dispatch over a fixed-size f32 stack with zero
// per-pixel heap allocation.
//
// Stack depth
// ~~~~~~~~~~~
// MAX_STACK_DEPTH is the maximum number of f32 values live on the evaluation
// stack at once.  For any binary expression tree of depth d the maximum stack
// depth is d+1.  Spyndex's most complex indices (EVI2, ARVI) have depth ~5,
// so 16 is a safe ceiling that fits entirely in a few cache lines.
//
// Performance expectation
// ~~~~~~~~~~~~~~~~~~~~~~~
// The per-instruction `match` branch and the inability to exploit formula
// structure (e.g. shared subexpressions) means this will be 15–30% slower
// than the hand-written normalized-difference kernel on simple formulas.
// For complex multi-band formulas (EVI, SAVI) the gap narrows because the
// fixed kernel setup cost is amortised over more instructions per pixel.

const MAX_STACK_DEPTH: usize = 16;

/// A single stack-machine instruction.
/// Band indices are positions into the band slice array supplied at execute
/// time, not raw rasterio band numbers — the Python side handles the mapping.
#[derive(Debug, Clone)]
enum Instruction {
    PushBand(usize),   // push bands[idx][pixel] as f32
    PushScalar(f32),   // push a compile-time constant
    Add,
    Sub,
    Mul,
    Div,               // pops b then a, pushes a / (b + EPSILON)
    Pow,               // pops b then a, pushes a.powf(b)
    Neg,               // negates top of stack
    Abs,               // absolute value of top of stack
    Sqrt,              // square root of top of stack
}

/// A compiled, executable expression.  Constructed once per index call via
/// `compile_expr`, then passed to `execute_expr_inplace` for each tile.
/// Exposed to Python as an opaque handle.
#[pyclass]
struct CompiledExpr {
    program:    Vec<Instruction>,
    num_bands:  usize,   // expected number of band arrays at execute time
}

/// Compile a post-order instruction list (produced by the Python-side parser)
/// into a `CompiledExpr`.
///
/// `instructions` is a list of dicts, each with a `"type"` key and optional
/// `"index"` (for PushBand) or `"value"` (for PushScalar) keys.
/// Example element: `{"type": "push_band", "index": 0}`
#[pyfunction]
fn compile_expr(instructions: Vec<Bound<'_, PyAny>>, num_bands: usize) -> PyResult<CompiledExpr> {
    let mut program = Vec::with_capacity(instructions.len());

    for instr in &instructions {
        let kind: String = instr.get_item("type")?.extract()?;
        let op = match kind.as_str() {
            "push_band" => {
                let idx: usize = instr.get_item("index")?.extract()?;
                if idx >= num_bands {
                    return Err(PyValueError::new_err(format!(
                        "Band index {idx} is out of range for num_bands={num_bands}."
                    )));
                }
                Instruction::PushBand(idx)
            }
            "push_scalar" => {
                let v: f32 = instr.get_item("value")?.extract()?;
                Instruction::PushScalar(v)
            }
            "add"  => Instruction::Add,
            "sub"  => Instruction::Sub,
            "mul"  => Instruction::Mul,
            "div"  => Instruction::Div,
            "pow"  => Instruction::Pow,
            "neg"  => Instruction::Neg,
            "abs"  => Instruction::Abs,
            "sqrt" => Instruction::Sqrt,
            other  => return Err(PyValueError::new_err(format!(
                "Unknown instruction type: {other:?}. \
                 Valid types: push_band, push_scalar, add, sub, mul, div, pow, neg, abs, sqrt."
            ))),
        };
        program.push(op);
    }

    // Static stack-depth check: walk the program once and verify the stack
    // never exceeds MAX_STACK_DEPTH and always has enough operands.
    let mut depth: isize = 0;
    for (i, instr) in program.iter().enumerate() {
        let delta: isize = match instr {
            Instruction::PushBand(_) | Instruction::PushScalar(_) => 1,
            Instruction::Add | Instruction::Sub |
            Instruction::Mul | Instruction::Div | Instruction::Pow => -1,
            Instruction::Neg | Instruction::Abs | Instruction::Sqrt => 0,
        };
        depth += delta;
        if depth <= 0 && !matches!(instr,
            Instruction::PushBand(_) | Instruction::PushScalar(_))
        {
            return Err(PyValueError::new_err(format!(
                "Stack underflow at instruction {i} ({instr:?}). \
                 Check that the bytecode is valid post-order."
            )));
        }
        if depth > MAX_STACK_DEPTH as isize {
            return Err(PyValueError::new_err(format!(
                "Expression requires stack depth {depth} which exceeds \
                 MAX_STACK_DEPTH={MAX_STACK_DEPTH}. Simplify the formula."
            )));
        }
    }
    if depth != 1 {
        return Err(PyValueError::new_err(format!(
            "Expression leaves {depth} values on the stack instead of 1. \
             Check that the bytecode is complete."
        )));
    }

    Ok(CompiledExpr { program, num_bands })
}

/// Execute a `CompiledExpr` element-wise across a set of 2-D band arrays,
/// writing results into `out_buffer`.
///
/// `bands` must be a list of uint16 arrays all sharing the same shape as
/// `out_buffer`, in the same order as the band indices used during compilation.
#[pyfunction]
fn execute_expr_inplace<'py>(
    py: Python<'py>,
    expr: Bound<'py, CompiledExpr>,
    bands: Vec<Bound<'py, PyArray2<u16>>>,
    out_buffer: Bound<'py, PyArray2<f32>>,
) -> PyResult<()> {
    let expr_ref = expr.borrow();

    // Validate band count
    if bands.len() != expr_ref.num_bands {
        return Err(PyValueError::new_err(format!(
            "Expected {} band arrays, got {}.",
            expr_ref.num_bands,
            bands.len()
        )));
    }

    // Validate all shapes match out_buffer
    let out_shape = out_buffer.shape().to_vec();
    for (i, band) in bands.iter().enumerate() {
        if band.shape() != out_shape.as_slice() {
            return Err(PyValueError::new_err(format!(
                "Band {i} shape {:?} does not match out_buffer shape {out_shape:?}.",
                band.shape()
            )));
        }
    }

    // Acquire read views for all bands
    let read_guards: Vec<_> = bands.iter().map(|b| b.readonly()).collect();
    let band_arrays: Vec<ndarray::ArrayView2<u16>> =
        read_guards.iter().map(|g| g.as_array()).collect();

    let mut out_write = out_buffer.readwrite();
    let mut out_view = out_write.as_array_mut();

    // Clone program and band count so they can be moved into the closure.
    // The program Vec is small (< 100 instructions for any real index) so
    // cloning is negligible relative to the pixel-loop cost.
    let program   = expr_ref.program.clone();

    // Inline the pixel evaluation logic as a macro so it can be shared between
    // the contiguous and non-contiguous paths without duplicating 30 lines.
    // Both paths are identical in logic; they differ only in how they index
    // into the output array and band arrays.
    macro_rules! eval_pixel {
        ($out_val:expr, $pixel_idx:expr, $band_slices:expr) => {{
            let mut stack = [0.0f32; MAX_STACK_DEPTH];
            let mut sp: usize = 0;
            for instr in &program {
                match instr {
                    Instruction::PushBand(idx) => {
                        stack[sp] = $band_slices[*idx][$pixel_idx] as f32;
                        sp += 1;
                    }
                    Instruction::PushScalar(v) => {
                        stack[sp] = *v;
                        sp += 1;
                    }
                    Instruction::Add  => { sp -= 1; stack[sp - 1] += stack[sp]; }
                    Instruction::Sub  => { sp -= 1; stack[sp - 1] -= stack[sp]; }
                    Instruction::Mul  => { sp -= 1; stack[sp - 1] *= stack[sp]; }
                    Instruction::Div  => { sp -= 1; stack[sp - 1] /= stack[sp] + EPSILON; }
                    Instruction::Pow  => { sp -= 1; stack[sp - 1] = stack[sp - 1].powf(stack[sp]); }
                    Instruction::Neg  => { stack[sp - 1] = -stack[sp - 1]; }
                    Instruction::Abs  => { stack[sp - 1] = stack[sp - 1].abs(); }
                    Instruction::Sqrt => { stack[sp - 1] = stack[sp - 1].sqrt(); }
                }
            }
            let val = stack[0];
            #[cfg(feature = "benchmark")]
            { $out_val = std::hint::black_box(val); }
            #[cfg(not(feature = "benchmark"))]
            { $out_val = val; }
        }};
    }

    // Check contiguity BEFORE consuming out_view.
    // is_standard_layout() returns true when the array is C-contiguous,
    // which lets us safely call as_slice() / as_slice_mut() on both the
    // output and all band arrays without moving any of them.
    let all_contiguous = out_view.is_standard_layout()
        && band_arrays.iter().all(|a| a.is_standard_layout());

    py.allow_threads(move || {
        if all_contiguous {
            // Fast path: flat slices, no ndarray indexing overhead.
            // as_slice_mut() is safe here because is_standard_layout() was
            // true above — the array is C-contiguous and uniquely owned via
            // the readwrite() guard acquired before allow_threads.
            let o_s = out_view
                .into_slice()
                .expect("out_view lost contiguity between check and use — should not happen.");

            let band_slices: Vec<&[u16]> = band_arrays
                .iter()
                .map(|a| a.as_slice().expect(
                    "Band array lost contiguity between check and use — should not happen."
                ))
                .collect();

            o_s.par_iter_mut()
                .enumerate()
                .for_each(|(i, out_val)| {
                    eval_pixel!(*out_val, i, band_slices);
                });

        } else {
            // Non-contiguous fallback (strided tiles from rasterio).
            // Collect flat owned copies of each band so we can index by
            // linear position consistently with the contiguous path.
            // This allocation is acceptable because non-contiguous layouts
            // are rare in practice.
            let band_slices: Vec<Vec<u16>> = band_arrays
                .iter()
                .map(|a| a.iter().copied().collect())
                .collect();

            // Convert band_slices to &[u16] slices for the macro
            let band_slices: Vec<&[u16]> = band_slices
                .iter()
                .map(|v| v.as_slice())
                .collect();

            let chunk_size = (out_view.nrows() / rayon::current_num_threads()).max(64);
            let ncols      = out_view.ncols();

            out_view
                .axis_chunks_iter_mut(Axis(0), chunk_size)
                .into_par_iter()
                .enumerate()
                .for_each(|(chunk_idx, mut o_chunk)| {
                    let row_offset = chunk_idx * chunk_size;
                    Zip::indexed(&mut o_chunk).for_each(|(row, col), out_val| {
                        let i = (row_offset + row) * ncols + col;
                        eval_pixel!(*out_val, i, band_slices);
                    });
                });
        }
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// Python-exposed functions (hand-written kernels)
// ---------------------------------------------------------------------------

/// In-place normalized difference using floating-point division.
///
/// Prefer this for tiles smaller than ~2048×2048.  Uses no extra RAM beyond
/// the caller-supplied output buffer.
#[pyfunction]
pub fn calculate_normalized_difference_inplace<'py>(
    py: Python<'py>,
    band_a: Bound<'py, PyArray2<u16>>,
    band_b: Bound<'py, PyArray2<u16>>,
    out_buffer: Bound<'py, PyArray2<f32>>,
) -> PyResult<()> {
    validate_shapes_inplace(band_a.shape(), band_b.shape(), out_buffer.shape())?;

    let a_view   = band_a.readonly();
    let b_view   = band_b.readonly();
    let a_arr    = a_view.as_array();
    let b_arr    = b_view.as_array();
    // readwrite() enforces exclusive access at the type level, making the
    // mutable borrow sound even across the py.allow_threads boundary.
    let mut out_write = out_buffer.readwrite();
    let out_view = out_write.as_array_mut();

    py.allow_threads(move || compute_logic(a_arr, b_arr, out_view));
    Ok(())
}

/// In-place normalized difference using a precomputed reciprocal LUT.
///
/// The LUT (~512 KB) is initialised once at module load time and shared across
/// all calls.  Typically 3–5% faster than the standard path for tiles >=
/// ~2048×2048.  For smaller tiles the LUT causes L2/L3 cache pressure and
/// will be slightly slower — use `calculate_normalized_difference_inplace`
/// instead.
#[pyfunction]
pub fn calculate_normalized_difference_lut_inplace<'py>(
    py: Python<'py>,
    band_a: Bound<'py, PyArray2<u16>>,
    band_b: Bound<'py, PyArray2<u16>>,
    out_buffer: Bound<'py, PyArray2<f32>>,
) -> PyResult<()> {
    validate_shapes_inplace(band_a.shape(), band_b.shape(), out_buffer.shape())?;

    let a_view   = band_a.readonly();
    let b_view   = band_b.readonly();
    let a_arr    = a_view.as_array();
    let b_arr    = b_view.as_array();
    let mut out_write = out_buffer.readwrite();
    let out_view = out_write.as_array_mut();

    py.allow_threads(move || compute_logic_lut(a_arr, b_arr, out_view));
    Ok(())
}

/// Allocating normalized difference.  Allocates and returns a new f32 array.
///
/// Uses the standard division path (no LUT).  Prefer the inplace variants for
/// pipeline use where the output buffer can be reused across tiles.
#[pyfunction]
pub fn calculate_normalized_difference<'py>(
    py: Python<'py>,
    band_a: Bound<'py, PyArray2<u16>>,
    band_b: Bound<'py, PyArray2<u16>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    validate_shapes(band_a.shape(), band_b.shape())?;

    let a_view = band_a.readonly();
    let b_view = band_b.readonly();
    let a_arr  = a_view.as_array();
    let b_arr  = b_view.as_array();

    let dims       = band_a.shape();
    let out_buffer = PyArray2::<f32>::zeros_bound(py, [dims[0], dims[1]], false);

    // readwrite() is sound here: out_buffer was just created above and no
    // other Python code can hold a reference to it yet.
    let mut out_write = out_buffer.readwrite();
    let out_view  = out_write.as_array_mut();

    py.allow_threads(move || compute_logic(a_arr, b_arr, out_view));
    Ok(out_buffer)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn geodium(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init_hardware_optimization();
    // Hand-written kernels
    m.add_function(wrap_pyfunction!(calculate_normalized_difference_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_normalized_difference_lut_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_normalized_difference, m)?)?;
    // Expression evaluator
    m.add_function(wrap_pyfunction!(compile_expr, m)?)?;
    m.add_function(wrap_pyfunction!(execute_expr_inplace, m)?)?;
    m.add_class::<CompiledExpr>()?;
    Ok(())
}