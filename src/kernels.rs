//! NVRTC compilation of the kernel blob and the per-function handle table.

use crate::error::{Error, Result};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};

pub(crate) type CUptr = cudarc::driver::sys::CUdeviceptr;

/// bf16 + f16 instantiation pair of one typed kernel.
pub(crate) struct HalfKernel {
    pub bf16: CudaFunction,
    pub f16: CudaFunction,
}

impl HalfKernel {
    pub(crate) fn get(&self, dt: crate::Dtype) -> &CudaFunction {
        match dt {
            crate::Dtype::Bf16 => &self.bf16,
            crate::Dtype::F16 => &self.f16,
            crate::Dtype::F32 => unreachable!("HalfKernel has no f32 variant"),
        }
    }
}

/// Compiled kernel handles + fixed scratch allocations.
///
/// Scratch buffers are allocated once at construction on the engine's
/// stream; the split-K/M/N dispatch gates are capped to these sizes
/// (`SPLITK_SCRATCH_CAP` / transpose cap in dispatch.rs).
pub(crate) struct Kernels {
    // f32 triad
    pub sgemm_nn: CudaFunction,
    pub sgemm_tn: CudaFunction,
    pub sgemm_nt: CudaFunction,
    pub sgemm_nn_slim: CudaFunction,
    pub sgemm_tn_slim: CudaFunction,
    pub sgemm_nt_slim: CudaFunction,
    pub sgemm_nn_gemv: CudaFunction,
    pub sgemm_tn_gemv: CudaFunction,
    pub sgemm_nt_gemv: CudaFunction,
    pub sgemm_nn_ultra_thin: CudaFunction,
    pub sgemm_nn_narrow: CudaFunction,
    pub sgemm_nn_narrow_small: CudaFunction,
    pub sgemm_tn_narrow: CudaFunction,
    pub sgemm_nt_narrow: CudaFunction,
    pub sgemm_nn_splitk32_partial: CudaFunction,
    pub sgemm_nn_splitk_slim_partial: CudaFunction,
    pub sgemm_tn_splitm_partial: CudaFunction,
    pub sgemm_splitk_reduce: CudaFunction,
    pub sgemm_splitm_reduce: CudaFunction,
    pub sgemm_dx_col_gemv: CudaFunction,
    pub sgemm_transpose_f32_2d: CudaFunction,
    // typed scalar tier
    pub sgemm_nn_gemv_typed: HalfKernel,
    pub sgemm_tn_gemv_typed: HalfKernel,
    pub sgemm_nt_gemv_typed: HalfKernel,
    pub sgemm_nn_ultra_thin_typed: HalfKernel,
    pub sgemm_nn_narrow_typed: HalfKernel,
    pub sgemm_nn_narrow_small_typed: HalfKernel,
    pub sgemm_tn_narrow_typed: HalfKernel,
    pub sgemm_nt_narrow_typed: HalfKernel,
    pub sgemm_nn_big_typed: HalfKernel,
    pub sgemm_tn_big_typed: HalfKernel,
    pub sgemm_nt_big_typed: HalfKernel,
    // tensor-core tier
    pub sgemm_nn_tc_typed: HalfKernel,
    pub sgemm_tn_tc_typed: HalfKernel,
    pub sgemm_nt_tc_typed: HalfKernel,
    pub sgemm_nn_tc64_typed: HalfKernel,
    pub sgemm_tn_tc64_typed: HalfKernel,
    pub sgemm_nt_tc64_typed: HalfKernel,
    // upcast-fallback casts
    pub cast_f32_to_bf16: CudaFunction,
    pub cast_bf16_to_f32: CudaFunction,
    pub cast_f32_to_f16: CudaFunction,
    pub cast_f16_to_f32: CudaFunction,
    // fixed scratch (raw pointers cached at alloc; buffers kept alive here)
    pub splitk_scratch_ptr: CUptr,
    pub transpose_scratch_ptr: CUptr,
    _splitk_scratch: cudarc::driver::CudaSlice<f32>,
    _transpose_scratch: cudarc::driver::CudaSlice<f32>,
    /// Keeps the compiled module loaded for the lifetime of the handles.
    _module: Arc<CudaModule>,
}

/// Map a CUDA compute capability to the NVRTC arch string.
pub(crate) fn nvrtc_arch(cc: (u32, u32)) -> &'static str {
    match cc {
        (12, _) => "sm_120",
        (10, _) => "sm_100",
        (9, _) => "sm_90",
        (8, 9) => "sm_89",
        (8, 6) => "sm_86",
        (8, 0) => "sm_80",
        _ => {
            if cc.0 > 12 {
                "sm_120"
            } else {
                // typed/TC tiers need sm_80+ (cp.async, ldmatrix, bf16 mma);
                // the f32 triad itself is portable further back.
                "sm_80"
            }
        }
    }
}

fn cuda_include_paths() -> Vec<String> {
    let mut candidates: Vec<String> = Vec::new();
    for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        if let Ok(p) = std::env::var(var) {
            candidates.push(format!("{p}/include"));
        }
    }
    for std_path in [
        "/usr/local/cuda/include",
        "/usr/local/cuda-13.2/include",
        "/usr/local/cuda-12.8/include",
        "/usr/local/cuda-12.6/include",
        "/usr/local/cuda-12.4/include",
        "/opt/cuda/include",
    ] {
        candidates.push(std_path.to_string());
    }
    candidates
        .into_iter()
        .filter(|p| std::path::Path::new(p).join("cuda_fp16.h").exists())
        .collect()
}

impl Kernels {
    pub(crate) fn compile(
        ctx: &Arc<CudaContext>,
        stream: &Arc<cudarc::driver::CudaStream>,
        arch: &'static str,
    ) -> Result<Self> {
        let sources = [
            include_str!("../kernels/prelude.cuh"),
            include_str!("../kernels/casts.cu"),
            include_str!("../kernels/sgemm_bi.cu"),
        ];
        let combined: String = sources.join("\n");

        // No --use_fast_math: approximate intrinsics and denormal flushing
        // would change bits. --fmad=true keeps fused multiply-add (the
        // kernels pin reduction FMAs explicitly via __fmaf_rn anyway).
        // SGB_GROUP_M: L2-swizzle row-group size — bit-exact across values
        // (only the CTA emission order changes, never a C[m,n] reduction
        // order), tuned per L2 size.
        let group_m: usize = match arch {
            "sm_80" | "sm_86" | "sm_87" => 8,
            _ => 16,
        };
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            options: vec![
                "--fmad=true".to_string(),
                "--extra-device-vectorization".to_string(),
                format!("-DSGB_GROUP_M={group_m}"),
            ],
            include_paths: cuda_include_paths(),
            ..Default::default()
        };

        let ptx = cudarc::nvrtc::compile_ptx_with_opts(combined, opts)
            .map_err(|e| Error::Cuda(format!("NVRTC compile failed: {e:?}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| Error::Cuda(format!("module load failed: {e:?}")))?;

        let get = |name: &str| -> Result<CudaFunction> {
            module
                .load_function(name)
                .map_err(|e| Error::Cuda(format!("load_function({name}): {e:?}")))
        };
        let load_half = |base: &str| -> Result<HalfKernel> {
            Ok(HalfKernel {
                bf16: get(&format!("{base}_bf16"))?,
                f16: get(&format!("{base}_f16"))?,
            })
        };
        let set_dynsmem = |f: &CudaFunction, bytes: i32, name: &str| -> Result<()> {
            f.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes,
            )
            .map_err(|e| Error::Cuda(format!("set MAX_DYNAMIC_SHARED for {name}: {e:?}")))
        };
        let load_half_dynsmem = |base: &str, bytes: i32| -> Result<HalfKernel> {
            let k = load_half(base)?;
            set_dynsmem(&k.bf16, bytes, base)?;
            set_dynsmem(&k.f16, bytes, base)?;
            Ok(k)
        };
        let get_dynsmem = |name: &str, bytes: i32| -> Result<CudaFunction> {
            let f = get(name)?;
            set_dynsmem(&f, bytes, name)?;
            Ok(f)
        };

        // Fixed scratch: split-K/M/N partials (1<<23 f32 = 32 MB) and the
        // NT-via-transpose W^T staging (1<<22 f32 = 16 MB). Dispatch gates
        // cap against these exact sizes.
        let splitk_scratch = stream
            .alloc_zeros::<f32>(1 << 23)
            .map_err(|e| Error::Cuda(format!("splitk_scratch alloc: {e:?}")))?;
        let transpose_scratch = stream
            .alloc_zeros::<f32>(1 << 22)
            .map_err(|e| Error::Cuda(format!("transpose_scratch alloc: {e:?}")))?;
        let cached = |b: &cudarc::driver::CudaSlice<f32>| -> CUptr {
            use cudarc::driver::DevicePtr;
            let (p, _g) = b.device_ptr(stream);
            p
        };
        let splitk_scratch_ptr = cached(&splitk_scratch);
        let transpose_scratch_ptr = cached(&transpose_scratch);
        // The allocations above are stream-ordered; nothing else runs on
        // this stream yet, so first kernel use is ordered after the memsets.

        Ok(Self {
            sgemm_nn: get_dynsmem("sgemm_bi_nn", 34 * 1024)?,
            sgemm_tn: get_dynsmem("sgemm_bi_tn", 34 * 1024)?,
            sgemm_nt: get_dynsmem("sgemm_bi_nt", 34 * 1024)?,
            sgemm_nn_slim: get("sgemm_bi_nn_slim")?,
            sgemm_tn_slim: get("sgemm_bi_tn_slim")?,
            sgemm_nt_slim: get("sgemm_bi_nt_slim")?,
            sgemm_nn_gemv: get("sgemm_bi_nn_gemv")?,
            sgemm_tn_gemv: get("sgemm_bi_tn_gemv")?,
            sgemm_nt_gemv: get("sgemm_bi_nt_gemv")?,
            sgemm_nn_ultra_thin: get("sgemm_bi_nn_ultra_thin")?,
            sgemm_nn_narrow: get("sgemm_bi_nn_narrow")?,
            sgemm_nn_narrow_small: get("sgemm_bi_nn_narrow_small")?,
            sgemm_tn_narrow: get("sgemm_bi_tn_narrow")?,
            sgemm_nt_narrow: get("sgemm_bi_nt_narrow")?,
            sgemm_nn_splitk32_partial: get("sgemm_bi_nn_splitk32_partial")?,
            sgemm_nn_splitk_slim_partial: get("sgemm_bi_nn_splitk_slim_partial")?,
            sgemm_tn_splitm_partial: get_dynsmem("sgemm_bi_tn_splitm_partial", 34 * 1024)?,
            sgemm_splitk_reduce: get("sgemm_bi_splitk_reduce")?,
            sgemm_splitm_reduce: get("sgemm_bi_splitm_reduce")?,
            sgemm_dx_col_gemv: get("sgemm_bi_dx_col_gemv")?,
            sgemm_transpose_f32_2d: get("sgemm_transpose_f32_2d")?,
            sgemm_nn_gemv_typed: load_half("sgemm_bi_nn_gemv")?,
            sgemm_tn_gemv_typed: load_half("sgemm_bi_tn_gemv")?,
            sgemm_nt_gemv_typed: load_half("sgemm_bi_nt_gemv")?,
            sgemm_nn_ultra_thin_typed: load_half("sgemm_bi_nn_ultra_thin")?,
            sgemm_nn_narrow_typed: load_half("sgemm_bi_nn_narrow")?,
            sgemm_nn_narrow_small_typed: load_half("sgemm_bi_nn_narrow_small")?,
            sgemm_tn_narrow_typed: load_half("sgemm_bi_tn_narrow")?,
            sgemm_nt_narrow_typed: load_half("sgemm_bi_nt_narrow")?,
            sgemm_nn_big_typed: load_half_dynsmem("sgemm_bi_nn_big", 34 * 1024)?,
            sgemm_tn_big_typed: load_half_dynsmem("sgemm_bi_tn_big", 34 * 1024)?,
            sgemm_nt_big_typed: load_half_dynsmem("sgemm_bi_nt_big", 34 * 1024)?,
            sgemm_nn_tc_typed: load_half_dynsmem("sgemm_bi_nn_tc", 75_776)?,
            sgemm_tn_tc_typed: load_half_dynsmem("sgemm_bi_tn_tc", 75_776)?,
            sgemm_nt_tc_typed: load_half_dynsmem("sgemm_bi_nt_tc", 75_776)?,
            sgemm_nn_tc64_typed: load_half("sgemm_bi_nn_tc64")?,
            sgemm_tn_tc64_typed: load_half("sgemm_bi_tn_tc64")?,
            sgemm_nt_tc64_typed: load_half("sgemm_bi_nt_tc64")?,
            cast_f32_to_bf16: get("sgb_cast_f32_to_bf16")?,
            cast_bf16_to_f32: get("sgb_cast_bf16_to_f32")?,
            cast_f32_to_f16: get("sgb_cast_f32_to_f16")?,
            cast_f16_to_f32: get("sgb_cast_f16_to_f32")?,
            splitk_scratch_ptr,
            transpose_scratch_ptr,
            _splitk_scratch: splitk_scratch,
            _transpose_scratch: transpose_scratch,
            _module: module,
        })
    }
}
