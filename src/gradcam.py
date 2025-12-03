"""
Grad-CAM adapted for 1D convolutional ECG models.

Generates a class activation map along the temporal axis to highlight
which ECG segments most influence a given prediction.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM1D:
    """
    Gradient-weighted Class Activation Mapping for 1D CNNs.

    Registers forward and backward hooks on the target convolutional layer,
    computes gradients w.r.t. the target class, and produces a 1D saliency map.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor: torch.Tensor,
                     target_class: int = None) -> np.ndarray:
        """
        Compute the Grad-CAM saliency map.

        Parameters
        ----------
        input_tensor : (1, n_leads, window) FloatTensor on appropriate device
        target_class : class index; if None, uses argmax of model output

        Returns
        -------
        cam : 1D ndarray of shape (window,), values in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)          # (1, n_classes)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        score.backward()

        # activations: (1, C, T)   gradients: (1, C, T)
        acts = self._activations[0]     # (C, T)
        grads = self._gradients[0]      # (C, T)

        weights = grads.mean(dim=-1)    # global average pooling over time: (C,)
        cam = (weights[:, None] * acts).sum(dim=0)   # (T,)
        cam = F.relu(cam)

        # upsample to original signal length
        cam = cam.cpu().numpy()
        T_orig = input_tensor.shape[-1]
        cam = np.interp(
            np.linspace(0, len(cam) - 1, T_orig),
            np.arange(len(cam)),
            cam
        )

        # normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def plot_gradcam_ecg(signal: np.ndarray, cam: np.ndarray,
                     true_label: str, pred_label: str,
                     fs: int = 360, lead_names: list = None,
                     save_path: str = None):
    """
    Overlay Grad-CAM heatmap on ECG signal.

    Parameters
    ----------
    signal   : (n_leads, window) float array
    cam      : (window,) saliency map, values in [0, 1]
    true_label, pred_label : string class names
    fs       : sampling frequency (Hz)
    """
    if lead_names is None:
        lead_names = [f'Lead {i+1}' for i in range(signal.shape[0])]

    n_leads = signal.shape[0]
    time_axis = np.arange(signal.shape[1]) / fs * 1000   # ms

    fig, axes = plt.subplots(n_leads, 1, figsize=(12, 3 * n_leads), sharex=True)
    if n_leads == 1:
        axes = [axes]

    colormap = cm.get_cmap('jet')

    for i, ax in enumerate(axes):
        sig = signal[i]

        # colour segments by CAM value
        for t in range(len(sig) - 1):
            colour = colormap(cam[t])
            ax.fill_between(time_axis[t:t+2], sig[t:t+2],
                            alpha=cam[t] * 0.7 + 0.1, color=colour)

        ax.plot(time_axis, sig, color='black', lw=0.8, alpha=0.9)
        ax.set_ylabel(lead_names[i], fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(
        f'Grad-CAM  |  True: {true_label}  |  Predicted: {pred_label}',
        fontsize=12, fontweight='bold'
    )
    axes[-1].set_xlabel('Time (ms)', fontsize=10)

    # colorbar
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=axes, label='Saliency', fraction=0.02, pad=0.04)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def demo_gradcam(model, sample_X: np.ndarray, sample_y: int,
                 target_layer, class_names: list,
                 device: torch.device, save_path: str = None):
    """End-to-end Grad-CAM demo for a single ECG beat."""
    model.eval()
    tensor = torch.FloatTensor(sample_X).unsqueeze(0).to(device)

    gcam = GradCAM1D(model, target_layer)
    cam = gcam.generate_cam(tensor, target_class=sample_y)
    gcam.remove_hooks()

    with torch.no_grad():
        pred = model(tensor).argmax(dim=1).item()

    fig = plot_gradcam_ecg(
        sample_X, cam,
        true_label=class_names[sample_y],
        pred_label=class_names[pred],
        lead_names=['MLII', 'V5'],
        save_path=save_path,
    )
    return fig, cam, pred
