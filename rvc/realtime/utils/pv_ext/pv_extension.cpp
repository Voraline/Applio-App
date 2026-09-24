#include <torch/extension.h>

torch::Tensor pv_synthesize_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor fade_out_sq,
    torch::Tensor fade_in_sq,
    torch::Tensor window_over_n,
    torch::Tensor t_over_n,
    torch::Tensor k_grid_delta,
    torch::Tensor phia,
    torch::Tensor absab
);

torch::Tensor phase_vocoder_synth(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor fade_out_sq,
    torch::Tensor fade_in_sq,
    torch::Tensor window_over_n,
    torch::Tensor k_grid_delta,
    torch::Tensor t_over_n,
    torch::Tensor phia,
    torch::Tensor absab
) {
    if (a.is_cuda()) {
        return pv_synthesize_cuda(
            a, b, fade_out_sq, fade_in_sq, window_over_n,
            t_over_n, k_grid_delta, phia, absab
        );
    }

    auto phase = torch::outer(t_over_n, k_grid_delta) + phia;
    auto synthesized = torch::mv(phase.cos(), absab);

    return a * fade_out_sq
         + b * fade_in_sq
         + synthesized * window_over_n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("phase_vocoder_synth", &phase_vocoder_synth, "Fused phase vocoder synthesis step.");
}
