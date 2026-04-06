use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use ndarray::Zip;
use rayon::ThreadPoolBuilder;

const EPSILON: f32 = 1e-10;

fn init_rayon() {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let _ = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();
}

fn check_shapes(shapes: &[&[usize]]) -> PyResult<()> {
    if shapes.is_empty() { return Ok(()); }
    let base = shapes[0];
    for (i, s) in shapes.iter().enumerate().skip(1) {
        if *s != base {
            return Err(PyValueError::new_err(format!(
                "Shape mismatch at index {i}: expected {base:?}, got {s:?}"
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

#[inline(always)]
fn nd_kernel(a: u16, b: u16) -> f32 {
    let fa = a as f32;
    let fb = b as f32;
    (fa - fb) / (fa + fb + EPSILON)
}

#[inline(always)]
fn ratio_kernel(a: u16, b: u16) -> f32 {
    (a as f32) / (b as f32 + EPSILON)
}

#[inline(always)]
fn adjusted_diff_kernel(a: u16, b: u16, l: f32) -> f32 {
    let fa = a as f32;
    let fb = b as f32;
    ((fa - fb) / (fa + fb + l + EPSILON)) * (1.0 + l)
}

#[inline(always)]
fn evi_kernel(nir: u16, red: u16, blue: u16, g: f32, c1: f32, c2: f32, l: f32) -> f32 {
    let n = nir as f32;
    let r = red as f32;
    let b = blue as f32;
    g * ((n - r) / (n + c1 * r - c2 * b + l + EPSILON))
}

// ---------------------------------------------------------------------------
// Inplace API
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn calculate_normalized_difference_inplace<'py>(
    py: Python<'py>,
    band_a: Bound<'py, PyArray2<u16>>,
    band_b: Bound<'py, PyArray2<u16>>,
    out_buffer: Bound<'py, PyArray2<f32>>,
) -> PyResult<()> {
    check_shapes(&[band_a.shape(), band_b.shape(), out_buffer.shape()])?;
    let a_guard = band_a.readonly();
    let b_guard = band_b.readonly();
    let mut out_guard = out_buffer.readwrite();

    let a_view = a_guard.as_array();
    let b_view = b_guard.as_array();
    let out_view = out_guard.as_array_mut();

    py.allow_threads(move || {
        Zip::from(out_view)
            .and(a_view)
            .and(b_view)
            .par_for_each(|out, &a, &b| *out = nd_kernel(a, b));
    });
    Ok(())
}

#[pyfunction]
pub fn calculate_ratio_inplace<'py>(
    py: Python<'py>,
    band_a: Bound<'py, PyArray2<u16>>,
    band_b: Bound<'py, PyArray2<u16>>,
    out_buffer: Bound<'py, PyArray2<f32>>,
) -> PyResult<()> {
    check_shapes(&[band_a.shape(), band_b.shape(), out_buffer.shape()])?;
    let a_guard = band_a.readonly();
    let b_guard = band_b.readonly();
    let mut out_guard = out_buffer.readwrite();

    let a_view = a_guard.as_array();
    let b_view = b_guard.as_array();
    let out_view = out_guard.as_array_mut();

    py.allow_threads(move || {
        Zip::from(out_view)
            .and(a_view)
            .and(b_view)
            .par_for_each(|out, &a, &b| *out = ratio_kernel(a, b));
    });
    Ok(())
}

#[pyfunction]
pub fn calculate_adjusted_difference_inplace<'py>(
    py: Python<'py>,
    band_a: Bound<'py, PyArray2<u16>>,
    band_b: Bound<'py, PyArray2<u16>>,
    l: f32,
    out_buffer: Bound<'py, PyArray2<f32>>,
) -> PyResult<()> {
    check_shapes(&[band_a.shape(), band_b.shape(), out_buffer.shape()])?;
    let a_guard = band_a.readonly();
    let b_guard = band_b.readonly();
    let mut out_guard = out_buffer.readwrite();

    let a_view = a_guard.as_array();
    let b_view = b_guard.as_array();
    let out_view = out_guard.as_array_mut();

    py.allow_threads(move || {
        Zip::from(out_view)
            .and(a_view)
            .and(b_view)
            .par_for_each(|out, &a, &b| *out = adjusted_diff_kernel(a, b, l));
    });
    Ok(())
}

#[pyfunction]
pub fn calculate_evi_inplace<'py>(
    py: Python<'py>,
    nir: Bound<'py, PyArray2<u16>>,
    red: Bound<'py, PyArray2<u16>>,
    blue: Bound<'py, PyArray2<u16>>,
    g: f32, c1: f32, c2: f32, l: f32,
    out_buffer: Bound<'py, PyArray2<f32>>,
) -> PyResult<()> {
    check_shapes(&[nir.shape(), red.shape(), blue.shape(), out_buffer.shape()])?;
    let n_guard = nir.readonly();
    let r_guard = red.readonly();
    let b_guard = blue.readonly();
    let mut out_guard = out_buffer.readwrite();

    let n_view = n_guard.as_array();
    let r_view = r_guard.as_array();
    let b_view = b_guard.as_array();
    let out_view = out_guard.as_array_mut();

    py.allow_threads(move || {
        Zip::from(out_view)
            .and(n_view)
            .and(r_view)
            .and(b_view)
            .par_for_each(|out, &n, &r, &b| *out = evi_kernel(n, r, b, g, c1, c2, l));
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// Expression Engine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum Instruction {
    PushBand(usize), PushScalar(f32),
    Add, Sub, Mul, Div, Pow, Neg, Abs, Sqrt,
}

#[pyclass]
pub struct CompiledExpr {
    program: Vec<Instruction>,
    num_bands: usize,
}

#[pyfunction]
fn compile_expr(instructions: Vec<Bound<'_, PyAny>>, num_bands: usize) -> PyResult<CompiledExpr> {
    let mut program = Vec::with_capacity(instructions.len());
    for instr in &instructions {
        let kind: String = instr.get_item("type")?.extract()?;
        program.push(match kind.as_str() {
            "push_band" => Instruction::PushBand(instr.get_item("index")?.extract()?),
            "push_scalar" => Instruction::PushScalar(instr.get_item("value")?.extract()?),
            "add" => Instruction::Add, "sub" => Instruction::Sub,
            "mul" => Instruction::Mul, "div" => Instruction::Div,
            "pow" => Instruction::Pow, "neg" => Instruction::Neg,
            "abs" => Instruction::Abs, "sqrt" => Instruction::Sqrt,
            _ => return Err(PyValueError::new_err(format!("Unknown op: {kind}"))),
        });
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
    if bands.len() != expr_ref.num_bands { return Err(PyValueError::new_err("Band count mismatch.")); }
    
    let band_guards: Vec<_> = bands.iter().map(|b| b.readonly()).collect();
    let mut out_guard = out_buffer.readwrite();

    let band_views: Vec<_> = band_guards.iter().map(|g| g.as_array()).collect();
    let out_view = out_guard.as_array_mut();
    let program = expr_ref.program.clone();

    py.allow_threads(move || {
        // FIXED: Zip::indexed(view) is the correct starting associated function
        Zip::indexed(out_view)
            .par_for_each(|(r, c), out_val| {
                let mut stack = [0.0f32; 16];
                let mut sp = 0;
                for instr in &program {
                    match instr {
                        Instruction::PushBand(i) => { 
                            stack[sp] = band_views[*i][[r, c]] as f32; 
                            sp += 1; 
                        }
                        Instruction::PushScalar(s) => { stack[sp] = *s; sp += 1; }
                        Instruction::Add => { sp -= 1; stack[sp-1] += stack[sp]; }
                        Instruction::Sub => { sp -= 1; stack[sp-1] -= stack[sp]; }
                        Instruction::Mul => { sp -= 1; stack[sp-1] *= stack[sp]; }
                        Instruction::Div => { sp -= 1; stack[sp-1] /= stack[sp] + EPSILON; }
                        Instruction::Pow => { sp -= 1; stack[sp-1] = stack[sp-1].powf(stack[sp]); }
                        Instruction::Neg => { stack[sp-1] = -stack[sp-1]; }
                        Instruction::Abs => { stack[sp-1] = stack[sp-1].abs(); }
                        Instruction::Sqrt => { stack[sp-1] = stack[sp-1].sqrt(); }
                    }
                }
                *out_val = stack[0];
            });
    });
    Ok(())
}

#[pyfunction]
pub fn calculate_normalized_difference<'py>(
    py: Python<'py>,
    band_a: Bound<'py, PyArray2<u16>>,
    band_b: Bound<'py, PyArray2<u16>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let shape = band_a.shape();
    let out = PyArray2::<f32>::zeros(py, [shape[0], shape[1]], false);
    // Use clone() to pass the Bound handles safely
    calculate_normalized_difference_inplace(py, band_a.clone(), band_b.clone(), out.clone())?;
    Ok(out)
}

#[pymodule]
fn geodium(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init_rayon();
    m.add_function(wrap_pyfunction!(calculate_normalized_difference_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_ratio_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_adjusted_difference_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_evi_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_normalized_difference, m)?)?;
    m.add_function(wrap_pyfunction!(compile_expr, m)?)?;
    m.add_function(wrap_pyfunction!(execute_expr_inplace, m)?)?;
    m.add_class::<CompiledExpr>()?;
    Ok(())
}