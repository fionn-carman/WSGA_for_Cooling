"""
Parity Plot Generator for XGBoost Regression Models

Creates publication-quality parity plots from saved model diagnostic data.
- Combines 40°C and 100°C data on same plot for temperature-dependent properties
- Handles log-transformed targets (shows both log and linear space)
- Includes 80% prediction intervals (filled region)
- Supports cascaded predictions for models with auxiliary features
- For Density: generates separate plots for conditional and cascaded predictions
- Proper units and formatting

Usage:
    python ParityPlot.py --property Thermal_Conductivity
    python ParityPlot.py --property Kinematic_Viscosity
    python ParityPlot.py --property Density
    python ParityPlot.py --property Density --show_cascaded
    python ParityPlot.py --property Heat_Capacity
    python ParityPlot.py --property BP-Measured
    python ParityPlot.py --all
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# Property Configuration
# ============================================================

PROPERTY_CONFIG = {
    'Thermal_Conductivity': {
        'targets': ['Thermal_Conductivity_40C', 'Thermal_Conductivity_100C'],
        'label': 'Thermal Conductivity',
        'unit': 'W m$^{-1}$K$^{-1}$',
        'log_transform': False,
    },
    'Kinematic_Viscosity': {
        'targets': ['Kinematic_Viscosity_40C', 'Kinematic_Viscosity_100C'],
        'label': 'Kinematic Viscosity',
        'unit': 'cSt',
        'log_transform': True,
    },
    'Density': {
        'targets': ['Density_40C_g_cm^3', 'Density_100C_g_cm^3'],
        'label': 'Density',
        'unit': 'g cm$^{-3}$',
        'log_transform': False,
    },
    'Heat_Capacity': {
        'targets': [
            'Heat_Capacity_Constant_Pressure_40C_J_K_Mol',
            'Heat_Capacity_Constant_Pressure_100C_J_K_Mol'
        ],
        'label': 'Heat Capacity (Cp)',
        'unit': 'J K$^{-1}$mol$^{-1}$',
        'log_transform': True,
    },
    'BP-Measured': {
        'targets': ['BP-Measured'],
        'label': 'Boiling Point',
        'unit': '°C',
        'log_transform': False,
    },
    'MP-Measured': {
        'targets': ['MP-Measured'],
        'label': 'Melting Point',
        'unit': '°C',
        'log_transform': False,
    },
    'flashpoint': {
        'targets': ['flashpoint'],
        'label': 'Flash Point',
        'unit': 'K',
        'log_transform': False,
    },
    'DC_exp': {
        'targets': ['DC_exp'],
        'label': 'Dielectric Constant',
        'unit': '',
        'log_transform': False,
    },
}

TEMP_COLORS = {
    '40C': {'color': '#1f77b4', 'label': '40°C'},
    '100C': {'color': '#d62728', 'label': '100°C'},
    'single': {'color': '#2ca02c', 'label': None}
}


# ============================================================
# Data Loading
# ============================================================

def load_diagnostic_data(target_dir):
    """Load diagnostic data including cascaded predictions if available."""
    plots_dir = target_dir / 'plots'
    npz_files = list(plots_dir.glob('*_diagnostic_plot_data.npz'))
    json_files = list(plots_dir.glob('*_diagnostic_metrics.json'))

    if not npz_files or not json_files:
        raise FileNotFoundError(f"Diagnostic files not found in {plots_dir}")

    data = np.load(npz_files[0])
    with open(json_files[0], 'r') as f:
        metrics = json.load(f)

    result = {
        'y_true': data['y_true'],
        'y_pred': data['y_pred'],
        'metrics': metrics
    }

    if 'pi_lower' in data and 'pi_upper' in data:
        result['pi_lower'] = data['pi_lower']
        result['pi_upper'] = data['pi_upper']

    # Check for cascaded predictions (for models with auxiliary features)
    if 'y_pred_cascaded' in data:
        result['y_pred_cascaded'] = data['y_pred_cascaded']
        result['has_cascaded'] = True
    else:
        result['has_cascaded'] = False

    return result


def load_model_metadata(target_dir):
    """Load model metadata to check for auxiliary features and cascaded metrics."""
    model_file = target_dir / 'model' / 'xgb_model.joblib'
    if model_file.exists():
        import joblib
        model_data = joblib.load(model_file)
        return {
            'log_target': model_data.get('log_target', False),
            'auxiliary_feature': model_data.get('auxiliary_feature'),
            'auxiliary_feature_name': model_data.get('auxiliary_feature_name'),
            'cascaded_r2': model_data.get('cascaded_r2'),
            'cascaded_rmse': model_data.get('cascaded_rmse'),
            'cascaded_mae': model_data.get('cascaded_mae'),
            'test_r2': model_data.get('test_r2'),
            'test_rmse': model_data.get('test_rmse'),
            'test_mae': model_data.get('test_mae'),
        }
    return {'log_target': False}


def check_log_transform(target_dir):
    metadata = load_model_metadata(target_dir)
    return metadata.get('log_target', False)


# ============================================================
# Plotting Utilities
# ============================================================

def compute_metrics(y_true, y_pred):
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }


def create_parity_plot(
    data_list,
    property_label,
    unit,
    output_path,
    log_space=False,
    figsize=(6, 6),
    show_cascaded=False
):
    """
    Create parity plot with optional cascaded predictions overlay.
    
    If show_cascaded=True and cascaded data is available, shows:
    - Conditional predictions (filled circles) 
    - Cascaded predictions (open circles, same color)
    - Separate metrics for each
    """
    fig, ax = plt.subplots(figsize=figsize)

    all_true = np.concatenate([d['y_true'] for d in data_list])
    all_pred = np.concatenate([d['y_pred'] for d in data_list])
    
    # Include cascaded predictions in axis limits if present
    if show_cascaded:
        cascaded_preds = [d.get('y_pred_cascaded') for d in data_list 
                         if d.get('y_pred_cascaded') is not None]
        if cascaded_preds:
            all_pred = np.concatenate([all_pred] + cascaded_preds)

    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())
    margin = (max_val - min_val) * 0.08
    plot_min = min_val - margin
    plot_max = max_val + margin

    # Parity line
    ax.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        'k--',
        linewidth=1.5,
        zorder=1
    )

    metrics_blocks = []

    for d in data_list:
        y_true = d['y_true']
        y_pred = d['y_pred']
        color = d['color']
        label = d.get('temp_label')
        has_cascaded = d.get('y_pred_cascaded') is not None and show_cascaded

        # Prediction intervals
        if d.get('pi_lower') is not None:
            sort_idx = np.argsort(y_true)
            ax.fill_between(
                y_true[sort_idx],
                d['pi_lower'][sort_idx],
                d['pi_upper'][sort_idx],
                color=color,
                alpha=0.12,
                linewidth=0,
                zorder=2
            )

        # Plot conditional predictions (filled circles)
        scatter_label = label
        if has_cascaded:
            scatter_label = f"{label} (cond.)" if label else "Conditional"
        
        ax.scatter(
            y_true,
            y_pred,
            c=color,
            s=40,
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5,
            label=scatter_label,
            zorder=3
        )

        # Compute conditional metrics
        m = compute_metrics(y_true, y_pred)
        
        if has_cascaded:
            # Plot cascaded predictions (open circles)
            y_pred_cascaded = d['y_pred_cascaded']
            cascaded_label = f"{label} (casc.)" if label else "Cascaded"
            
            ax.scatter(
                y_true,
                y_pred_cascaded,
                facecolors='none',
                edgecolors=color,
                s=40,
                alpha=0.8,
                linewidth=1.5,
                label=cascaded_label,
                zorder=4,
                marker='o'
            )
            
            # Compute cascaded metrics
            m_cascaded = compute_metrics(y_true, y_pred_cascaded)
            
            header = label if label else "Model"
            metrics_blocks.append(
                f"{header}\n"
                f"Conditional:\n"
                f"  R² = {m['r2']:.4f}\n"
                f"  RMSE = {m['rmse']:.4f}\n"
                f"Cascaded:\n"
                f"  R² = {m_cascaded['r2']:.4f}\n"
                f"  RMSE = {m_cascaded['rmse']:.4f}"
            )
        else:
            header = label if label else "All Data"
            metrics_blocks.append(
                f"{header}\n"
                f"R²   = {m['r2']:.4f}\n"
                f"RMSE = {m['rmse']:.4f}\n"
                f"MAE  = {m['mae']:.4f}"
            )

    metrics_text = "\n\n".join(metrics_blocks)

    ax.text(
        0.05, 0.95, metrics_text,
        transform=ax.transAxes,
        fontsize=10 if not show_cascaded else 9,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            alpha=0.85,
            edgecolor='gray'
        )
    )

    unit_str = f' / {unit}' if unit else ''
    ax.set_xlabel(f'{property_label}{unit_str} — Experimental', fontsize=12)
    ax.set_ylabel(f'{property_label}{unit_str} — XGBoost', fontsize=12)

    # Show legend if multiple datasets or cascaded
    if len(data_list) > 1 or show_cascaded:
        ax.legend(loc='lower right', fontsize=9, frameon=False)

    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_aspect('equal')
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def create_single_parity_plot(
    data_list,
    property_label,
    unit,
    output_path,
    log_space=False,
    figsize=(6, 6),
    use_cascaded=False
):
    """
    Create parity plot using either conditional or cascaded predictions (not both).
    
    Args:
        use_cascaded: If True, use cascaded predictions; if False, use conditional predictions
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Select which predictions to use
    all_true = np.concatenate([d['y_true'] for d in data_list])
    if use_cascaded:
        all_pred = np.concatenate([d.get('y_pred_cascaded', d['y_pred']) for d in data_list])
    else:
        all_pred = np.concatenate([d['y_pred'] for d in data_list])

    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())
    margin = (max_val - min_val) * 0.08
    plot_min = min_val - margin
    plot_max = max_val + margin

    # Parity line
    ax.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        'k--',
        linewidth=1.5,
        zorder=1
    )

    metrics_blocks = []

    for d in data_list:
        y_true = d['y_true']
        
        # Select predictions based on mode
        if use_cascaded and d.get('y_pred_cascaded') is not None:
            y_pred = d['y_pred_cascaded']
        else:
            y_pred = d['y_pred']
        
        color = d['color']
        label = d.get('temp_label')

        # Prediction intervals
        if d.get('pi_lower') is not None:
            sort_idx = np.argsort(y_true)
            ax.fill_between(
                y_true[sort_idx],
                d['pi_lower'][sort_idx],
                d['pi_upper'][sort_idx],
                color=color,
                alpha=0.12,
                linewidth=0,
                zorder=2
            )

        # Plot predictions
        ax.scatter(
            y_true,
            y_pred,
            c=color,
            s=40,
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5,
            label=label,
            zorder=3
        )

        # Compute metrics
        m = compute_metrics(y_true, y_pred)
        
        header = label if label else "All Data"
        metrics_blocks.append(
            f"{header}\n"
            f"R²   = {m['r2']:.4f}\n"
            f"RMSE = {m['rmse']:.4f}\n"
            f"MAE  = {m['mae']:.4f}"
        )

    metrics_text = "\n\n".join(metrics_blocks)

    ax.text(
        0.05, 0.95, metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            alpha=0.85,
            edgecolor='gray'
        )
    )

    unit_str = f' / {unit}' if unit else ''
    ax.set_xlabel(f'{property_label}{unit_str} — Experimental', fontsize=12)
    ax.set_ylabel(f'{property_label}{unit_str} — XGBoost', fontsize=12)

    # Show legend if multiple datasets
    if len(data_list) > 1:
        ax.legend(loc='lower right', fontsize=9, frameon=False)

    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_aspect('equal')
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def create_residuals_plot(
    data_list,
    property_label,
    unit,
    output_path,
    log_space=False,
    figsize=(8, 6)
):
    """Create residuals plot."""
    fig, ax = plt.subplots(figsize=figsize)

    all_true = np.concatenate([d['y_true'] for d in data_list])
    margin = (all_true.max() - all_true.min()) * 0.05

    for d in data_list:
        y_true = d['y_true']
        y_pred = d['y_pred']
        residuals = y_pred - y_true

        color = d['color']
        label = d.get('temp_label')

        ax.scatter(
            y_true,
            residuals,
            c=color,
            s=40,
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5,
            label=label
        )

        # Optional: show 80% PI-derived residual band
        if d.get('pi_lower') is not None:
            sort_idx = np.argsort(y_true)
            lower_res = d['pi_lower'][sort_idx] - y_true[sort_idx]
            upper_res = d['pi_upper'][sort_idx] - y_true[sort_idx]

            ax.fill_between(
                y_true[sort_idx],
                lower_res,
                upper_res,
                color=color,
                alpha=0.12,
                linewidth=0
            )

    ax.axhline(0, color='k', linestyle='--', linewidth=1)

    unit_str = f' / {unit}' if unit else ''
    ax.set_xlabel(f'{property_label}{unit_str} — Experimental', fontsize=12)
    ax.set_ylabel(
        f'Residual ({property_label}$_{{pred}}$ − {property_label}$_{{exp}}$)',
        fontsize=12
    )

    if len(data_list) > 1 and any(d.get('temp_label') for d in data_list):
        ax.legend(frameon=False, fontsize=10)

    ax.set_xlim(all_true.min() - margin, all_true.max() + margin)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


# ============================================================
# Plot Generation
# ============================================================

def generate_parity_plots(base_dir, property_name, output_dir=None, show_cascaded=False):
    """Generate parity plots for a property."""
    base_dir = Path(base_dir)
    output_dir = Path(output_dir) if output_dir else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PROPERTY_CONFIG[property_name]
    targets = config['targets']

    data_list = []
    actual_log_transform = False

    for target in targets:
        target_dir = base_dir / target
        if not target_dir.exists():
            continue

        data = load_diagnostic_data(target_dir)
        metadata = load_model_metadata(target_dir)
        is_log = metadata.get('log_target', False)
        actual_log_transform |= is_log

        if '40C' in target:
            temp_key = '40C'
        elif '100C' in target:
            temp_key = '100C'
        else:
            temp_key = 'single'

        temp_info = TEMP_COLORS[temp_key]

        data_entry = {
            'y_true': data['y_true'],
            'y_pred': data['y_pred'],
            'pi_lower': data.get('pi_lower'),
            'pi_upper': data.get('pi_upper'),
            'color': temp_info['color'],
            'temp_label': temp_info['label'],
            'has_auxiliary': metadata.get('auxiliary_feature') is not None
        }
        
        # Add cascaded predictions if available
        if data.get('has_cascaded'):
            data_entry['y_pred_cascaded'] = data['y_pred_cascaded']
        
        data_list.append(data_entry)

    if not data_list:
        print(f"No data found for {property_name}")
        return

    # Check if any model has cascaded data
    any_cascaded = any(d.get('y_pred_cascaded') is not None for d in data_list)
    
    if actual_log_transform:
        # The npz file contains LINEAR-SPACE values (training script applies expm1 before saving)
        # So for the linear plot, use data as-is
        # For the log plot, apply log1p to convert back to log space
        
        # Linear space plot (data is already in linear space)
        create_parity_plot(
            data_list,
            config['label'],
            config['unit'],
            output_dir / f'{property_name}_parity_linear.png',
            log_space=False,
            show_cascaded=show_cascaded and any_cascaded
        )

        create_residuals_plot(
            data_list,
            config['label'],
            config['unit'],
            output_dir / f'{property_name}_residuals_linear.png',
            log_space=False
        )

        # Transform to log space for log plot
        log_data = []
        for d in data_list:
            ld = {
                'y_true': np.log1p(d['y_true']),
                'y_pred': np.log1p(d['y_pred']),
                'pi_lower': np.log1p(d['pi_lower']) if d['pi_lower'] is not None else None,
                'pi_upper': np.log1p(d['pi_upper']) if d['pi_upper'] is not None else None,
                'color': d['color'],
                'temp_label': d['temp_label']
            }
            if d.get('y_pred_cascaded') is not None:
                ld['y_pred_cascaded'] = np.log1p(d['y_pred_cascaded'])
            log_data.append(ld)

        # Log space plot
        create_parity_plot(
            log_data,
            config['label'],
            f'log({config["unit"]})' if config['unit'] else 'log',
            output_dir / f'{property_name}_parity_log.png',
            log_space=True,
            show_cascaded=show_cascaded and any_cascaded
        )
        
        create_residuals_plot(
            log_data,
            config['label'],
            f'log({config["unit"]})' if config['unit'] else 'log',
            output_dir / f'{property_name}_residuals_log.png',
            log_space=True
        )

    else:
        # For Density: generate separate conditional and cascaded plots
        if property_name == 'Density' and any_cascaded:
            # Conditional predictions plot
            create_single_parity_plot(
                data_list,
                config['label'],
                config['unit'],
                output_dir / f'{property_name}_parity_conditional.png',
                log_space=False,
                use_cascaded=False
            )
            
            # Cascaded predictions plot
            create_single_parity_plot(
                data_list,
                config['label'],
                config['unit'],
                output_dir / f'{property_name}_parity_cascaded.png',
                log_space=False,
                use_cascaded=True
            )
            
            # Also generate residuals plot (conditional only)
            create_residuals_plot(
                data_list,
                config['label'],
                config['unit'],
                output_dir / f'{property_name}_residuals.png',
                log_space=False
            )
        else:
            # Standard plot for other properties
            suffix = '_cascaded' if show_cascaded and any_cascaded else ''
            create_parity_plot(
                data_list,
                config['label'],
                config['unit'],
                output_dir / f'{property_name}_parity{suffix}.png',
                log_space=False,
                show_cascaded=show_cascaded and any_cascaded
            )
            
            create_residuals_plot(
                data_list,
                config['label'],
                config['unit'],
                output_dir / f'{property_name}_residuals.png',
                log_space=False
            )


def generate_all_parity_plots(base_dir, output_dir=None, show_cascaded=False):
    """Generate parity plots for all properties."""
    for property_name in PROPERTY_CONFIG:
        try:
            generate_parity_plots(base_dir, property_name, output_dir, show_cascaded)
        except FileNotFoundError as e:
            print(f"Skipping {property_name}: {e}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate parity plots for XGBoost regression models"
    )

    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--property", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./ParityPlots")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--show_cascaded", action="store_true",
                        help="Show cascaded predictions for models with auxiliary features")

    args = parser.parse_args()

    if args.all:
        generate_all_parity_plots(args.base_dir, args.output_dir, args.show_cascaded)
    elif args.property:
        generate_parity_plots(args.base_dir, args.property, args.output_dir, args.show_cascaded)
    else:
        print("Please specify --property or --all")