use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use ndarray::{Axis, Zip};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::sync::OnceLock;

const EPSILON: f32 = 1e-10;

static RECIPROCAL_LUT: OnceLock<Vec<f32>> = OnceLock::new();

fn init_hardware_optimization() {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let _ = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();

    RECIPROCAL_LUT.get_or_init(|| {
        (0u32..=131_070)
            .map(|i| 1.0_f32 / (i as f32 + EPSILON))
            .collect()
    });
}

// ---------------------------------------------------------------------------
// Shape validation
// ---------------------------------------------------------------------------

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
// Core compute kernels
// ---------------------------------------------------------------------------

fn compute_logic(
    a_arr: ndarray::ArrayView2<u16>,
    b_arr: ndarray::ArrayView2<u16>,
    mut out: ndarray::ArrayViewMut2<f32>,
) {
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

const MAX_STACK_DEPTH: usize = 16;

#[derive(Debug, Clone)]
enum Instruction {
    PushBand(usize),
    PushScalar(f32),
    Add, Sub, Mul, Div, Pow, Neg, Abs, Sqrt,
}

#[pyclass]
struct CompiledExpr {
    program:    Vec<Instruction>,
    num_bands:  usize,
}

#[pyfunction]
fn compile_expr(instructions: Vec<Bound<'_, PyAny>>, num_bands: usize) -> PyResult<CompiledExpr> {
    let mut program = Vec::with_capacity(instructions.len());

    for instr in &instructions {
        let kind: String = instr.get_item("type")?.extract()?;
        let op = match kind.as_str() {
            "push_band" => {
                let idx: usize = instr.get_item("index")?.extract()?;
                if idx >= num_bands {
                    return Err(PyValueError::new_err(format!("Band index {idx} out of range.")));
                }
                Instruction::PushBand(idx)
            }
            "push_scalar" => Instruction::PushScalar(instr.get_item("value")?.extract()?),
            "add"  => Instruction::Add,
            "sub"  => Instruction::Sub,
            "mul"  => Instruction::Mul,
            "div"  => Instruction::Div,
            "pow"  => Instruction::Pow,
            "neg"  => Instruction::Neg,
            "abs"  => Instruction::Abs,
            "sqrt" => Instruction::Sqrt,
            other  => return Err(PyValueError::new_err(format!("Unknown instruction: {other}"))),
        };
        program.push(op);
    }

    let mut depth: isize = 0;
    for (i, instr) in program.iter().enumerate() {
        let delta: isize = match instr {
            Instruction::PushBand(_) | Instruction::PushScalar(_) => 1,
            Instruction::Add | Instruction::Sub |
            Instruction::Mul | Instruction::Div | Instruction::Pow => -1,
            Instruction::Neg | Instruction::Abs | Instruction::Sqrt => 0,
        };
        depth += delta;
        if depth <= 0 && !matches!(instr, Instruction::PushBand(_) | Instruction::PushScalar(_)) {
            return Err(PyValueError::new_err(format!("Stack underflow at instr {i}")));
        }
    }
    if depth != 1 {
        return Err(PyValueError::new_err("Expression incomplete."));
    }

    Ok(CompiledExpr { program, num_bands })
}

#[pyfunction]
fn execute_expr_inplace<'py>(
    py: Python<'py>,
    expr: Bound<'py, CompiledExpr>,
    bands: Vec<Bound<'py, PyArray2<u16>>>,
    out_buffer: Bound<'py, PyArray2<f32>>,
) -> PyResult<()> {
    let expr_ref = expr.borrow();

    if bands.len() != expr_ref.num_bands {
        return Err(PyValueError::new_err("Band count mismatch."));
    }

    let out_shape = out_buffer.shape().to_vec();
    for band in &bands {
        if band.shape() != out_shape.as_slice() {
            return Err(PyValueError::new_err("Shape mismatch."));
        }
    }

    let read_guards: Vec<_> = bands.iter().map(|b| b.readonly()).collect();
    let band_arrays: Vec<ndarray::ArrayView2<u16>> =
        read_guards.iter().map(|g| g.as_array()).collect();

    let mut out_write = out_buffer.readwrite();
    let mut out_view = out_write.as_array_mut();
    let program = expr_ref.program.clone();

    macro_rules! eval_pixel {
        ($out_val:expr, $pixel_idx:expr, $band_slices:expr) => {{
            let mut stack = [0.0f32; MAX_STACK_DEPTH];
            let mut sp: usize = 0;
            for instr in &program {
                match instr {
                    Instruction::PushBand(idx) => { stack[sp] = $band_slices[*idx][$pixel_idx] as f32; sp += 1; }
                    Instruction::PushScalar(v) => { stack[sp] = *v; sp += 1; }
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
            #[cfg(feature = "benchmark")] { $out_val = std::hint::black_box(val); }
            #[cfg(not(feature = "benchmark"))] { $out_val = val; }
        }};
    }

    let all_contiguous = out_view.is_standard_layout()
        && band_arrays.iter().all(|a| a.is_standard_layout());

    py.allow_threads(move || {
        if all_contiguous {
            let o_s = out_view.into_slice().unwrap();
            let band_slices: Vec<&[u16]> = band_arrays.iter().map(|a| a.as_slice().unwrap()).collect();

            o_s.par_iter_mut().enumerate().for_each(|(i, out_val)| {
                eval_pixel!(*out_val, i, band_slices);
            });
        } else {
            let band_slices: Vec<Vec<u16>> = band_arrays.iter().map(|a| a.iter().copied().collect()).collect();
            let b_refs: Vec<&[u16]> = band_slices.iter().map(|v| v.as_slice()).collect();
            let chunk_size = (out_view.nrows() / rayon::current_num_threads()).max(64);
            let ncols = out_view.ncols();

            out_view.axis_chunks_iter_mut(Axis(0), chunk_size)
                .into_par_iter()
                .enumerate()
                .for_each(|(chunk_idx, mut o_chunk)| {
                    let row_offset = chunk_idx * chunk_size;
                    Zip::indexed(&mut o_chunk).for_each(|(row, col), out_val| {
                        let i = (row_offset + row) * ncols + col;
                        eval_pixel!(*out_val, i, b_refs);
                    });
                });
        }
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// Python-exposed functions
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn calculate_normalized_difference_inplace<'py>(
    py: Python<'py>,
    band_a: Bound<'py, PyArray2<u16>>,
    band_b: Bound<'py, PyArray2<u16>>,
    out_buffer: Bound<'py, PyArray2<f32>>,
) -> PyResult<()> {
    validate_shapes_inplace(band_a.shape(), band_b.shape(), out_buffer.shape())?;
    let a_view = band_a.readonly();
    let b_view = band_b.readonly();
    let a_arr = a_view.as_array();
    let b_arr = b_view.as_array();
    let mut out_write = out_buffer.readwrite();
    let out_view = out_write.as_array_mut();

    py.allow_threads(move || compute_logic(a_arr, b_arr, out_view));
    Ok(())
}

#[pyfunction]
pub fn calculate_normalized_difference_lut_inplace<'py>(
    py: Python<'py>,
    band_a: Bound<'py, PyArray2<u16>>,
    band_b: Bound<'py, PyArray2<u16>>,
    out_buffer: Bound<'py, PyArray2<f32>>,
) -> PyResult<()> {
    validate_shapes_inplace(band_a.shape(), band_b.shape(), out_buffer.shape())?;
    let a_view = band_a.readonly();
    let b_view = band_b.readonly();
    let a_arr = a_view.as_array();
    let b_arr = b_view.as_array();
    let mut out_write = out_buffer.readwrite();
    let out_view = out_write.as_array_mut();

    py.allow_threads(move || compute_logic_lut(a_arr, b_arr, out_view));
    Ok(())
}

#[pyfunction]
pub fn calculate_normalized_difference<'py>(
    py: Python<'py>,
    band_a: Bound<'py, PyArray2<u16>>,
    band_b: Bound<'py, PyArray2<u16>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    validate_shapes(band_a.shape(), band_b.shape())?;
    let a_view = band_a.readonly();
    let b_view = band_b.readonly();
    let a_arr = a_view.as_array();
    let b_arr = b_view.as_array();

    let dims = band_a.shape();
    // CHANGED: zeros_bound -> zeros
    let out_buffer = PyArray2::<f32>::zeros(py, [dims[0], dims[1]], false);
    
    let mut out_write = out_buffer.readwrite();
    let out_view = out_write.as_array_mut();

    py.allow_threads(move || compute_logic(a_arr, b_arr, out_view));
    Ok(out_buffer)
}

#[pymodule]
fn geodium(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init_hardware_optimization();
    m.add_function(wrap_pyfunction!(calculate_normalized_difference_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_normalized_difference_lut_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_normalized_difference, m)?)?;
    m.add_function(wrap_pyfunction!(compile_expr, m)?)?;
    m.add_function(wrap_pyfunction!(execute_expr_inplace, m)?)?;
    m.add_class::<CompiledExpr>()?;
    Ok(())
}