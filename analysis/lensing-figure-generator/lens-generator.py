#!/usr/bin/env python3
"""
Generate multiple static versions of the lensing diagram
representing different interactive states for publication
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import argparse

import matplotlib
import matplotlib.gridspec as gridspec
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Ellipse, Wedge

DEFAULT_STYLE = {
    "font.family": "serif",
    "font.size": 12,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
}


def render_angle_indicator(
    ax,
    *,
    center: Tuple[float, float],
    radius: float,
    theta1: float,
    theta2: float,
    color: str,
    label: str,
    label_offset: float = 0.2,
    wedge_width: float = 0.12,
    wedge_alpha: float = 0.25,
    zorder: int = 6,
) -> None:
    """Render an angular indicator with a highlighted arc and label."""
    theta_start, theta_end = sorted([theta1, theta2])
    if np.isclose(theta_start, theta_end):
        return

    wedge_width = min(wedge_width, radius * 0.9)

    wedge = Wedge(
        center,
        radius,
        theta_start,
        theta_end,
        width=wedge_width,
        facecolor=color,
        alpha=wedge_alpha,
        edgecolor='none',
        zorder=zorder,
    )
    ax.add_patch(wedge)

    arc = Arc(
        center,
        2 * radius,
        2 * radius,
        angle=0,
        theta1=theta_start,
        theta2=theta_end,
        color=color,
        linewidth=1.6,
        zorder=zorder + 1,
    )
    ax.add_patch(arc)

    label_angle = np.radians((theta_start + theta_end) / 2.0)
    label_radius = radius + label_offset
    label_x = center[0] + label_radius * np.cos(label_angle)
    label_y = center[1] + label_radius * np.sin(label_angle)

    ha = 'left' if np.cos(label_angle) >= 0 else 'right'
    va = 'bottom' if np.sin(label_angle) >= 0 else 'top'

    ax.text(
        label_x,
        label_y,
        label,
        color=color,
        fontsize=plt.rcParams.get("font.size", 12),
        ha=ha,
        va=va,
        bbox=dict(boxstyle="round,pad=0.25", facecolor='white', alpha=0.85, edgecolor='none'),
        zorder=zorder + 2,
    )


@dataclass(frozen=True)
class SnapshotConfig:
    """Configuration describing an individual lensing snapshot."""

    source_offset: float
    panel_title: str
    individual_title: str
    filename_prefix: str
    scenario: str = "auto"


def default_snapshot_configs() -> Tuple[SnapshotConfig, ...]:
    """Return the default set of configurations used across the script."""
    return (
        SnapshotConfig(
            source_offset=0.0,
            panel_title="(a) Perfect Alignment: Einstein Ring",
            individual_title="Perfect Alignment - Einstein Ring",
            filename_prefix="lensing_einstein_ring",
            scenario="einstein_ring",
        ),
        SnapshotConfig(
            source_offset=0.3,
            panel_title="(b) Small Offset: Tangential Arc",
            individual_title="Small Source Offset - Tangential Arc",
            filename_prefix="lensing_arc_offset",
            scenario="arc",
        ),
        SnapshotConfig(
            source_offset=0.8,
            panel_title="(c) Large Offset: Two Images",
            individual_title="Large Source Offset - Double Images",
            filename_prefix="lensing_double_offset",
            scenario="double",
        ),
        SnapshotConfig(
            source_offset=0.2,
            panel_title="(d) Source Inside Caustic: Four Images",
            individual_title="Quadruply Lensed Source",
            filename_prefix="lensing_quadruple",
            scenario="quad",
        ),
    )

def create_lensing_snapshot(
    ax,
    *,
    source_offset: float = 0.6,
    einstein_radius: float = 1.0,
    show_rays: bool = True,
    show_angles: bool = True,
    title: str = "",
    scenario: str | None = None,
):
    """
    Create a single lensing diagram with specified parameters.
    """
    ax.set_xlim(-4, 4)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    base_font = plt.rcParams.get("font.size", 12)
    title_font = base_font * 1.1
    annotation_font = base_font * 0.95

    ray_style = dict(arrowstyle='->', color='#FF8C42', linewidth=1.8, alpha=0.9)

    # Positions
    obs_x = -3.2
    lens_x = 0
    source_x = 3
    source_y = source_offset

    # Draw planes
    plane_style = dict(color='gray', alpha=0.3, linewidth=0.8)
    ax.axvline(x=lens_x, ymin=0.15, ymax=0.85, **plane_style)
    ax.axvline(x=source_x, ymin=0.15, ymax=0.85, **plane_style)

    # Labels
    ax.text(lens_x, 2.0, r'Lens plane', ha='center', fontsize=annotation_font, color='gray')
    ax.text(source_x, 2.0, r'Source plane', ha='center', fontsize=annotation_font, color='gray')

    # Observer
    observer = Circle((obs_x, 0), 0.06, color='black', zorder=5)
    ax.add_patch(observer)
    ax.text(obs_x, -0.25, 'Observer', ha='center', fontsize=annotation_font)

    # Lens mass
    lens_mass = Ellipse((lens_x, 0.1), width=0.32, height=0.20,
                        facecolor='purple', alpha=0.4,
                        edgecolor='purple', linewidth=2, zorder=3)
    ax.add_patch(lens_mass)

    # Einstein radius marker
    einstein_ring = Circle((lens_x, 0), einstein_radius,
                           fill=False, edgecolor='blue',
                           linewidth=2, alpha=0.7, linestyle='--')
    ax.add_patch(einstein_ring)
    ax.text(0.0, einstein_radius + 0.15, r'$\theta_\mathrm{E}$',
            color='blue', ha='center', fontsize=base_font)

    # Source
    source = Circle((source_x, source_y), 0.05, color='red', zorder=5)
    ax.add_patch(source)
    ax.text(source_x + 0.15, source_y + 0.1, 'Source',
            color='red', ha='left', fontsize=annotation_font)

    scenario = (scenario or ("einstein_ring" if abs(source_offset) < 0.05 else "double")).lower()
    image_positions: list[tuple[float, float]] = []

    if scenario == "einstein_ring":
        ring = Circle((lens_x, 0), einstein_radius,
                      fill=False, edgecolor='red',
                      linewidth=3, alpha=0.6, zorder=4)
        ax.add_patch(ring)
        ax.text(lens_x + einstein_radius + 0.2, 0.1,
                'Einstein Ring', color='red', fontsize=annotation_font, style='italic')
        image_positions.extend([(lens_x, einstein_radius), (lens_x, -einstein_radius)])

        if show_rays:
            for hit_y in (einstein_radius, -einstein_radius):
                ax.annotate('', xy=(lens_x, hit_y), xytext=(source_x, source_y),
                            arrowprops=ray_style)
                ax.annotate('', xy=(obs_x, 0), xytext=(lens_x, hit_y),
                            arrowprops=ray_style)

    elif scenario == "arc":
        arc_theta1, arc_theta2 = 40, 140
        arc_wedge = Wedge((lens_x, 0), einstein_radius, arc_theta1, arc_theta2,
                          width=0.18, facecolor='red', edgecolor='red',
                          alpha=0.55, zorder=4)
        ax.add_patch(arc_wedge)
        arc_mid = np.radians((arc_theta1 + arc_theta2) / 2)
        arc_point = (lens_x + (einstein_radius - 0.05) * np.cos(arc_mid),
                     (einstein_radius - 0.05) * np.sin(arc_mid))
        image_positions.append((lens_x + einstein_radius * np.cos(arc_mid),
                                einstein_radius * np.sin(arc_mid)))

        counter_y = -0.35
        counter = Circle((lens_x, counter_y), 0.035,
                         color='red', alpha=0.45, zorder=4)
        ax.add_patch(counter)
        image_positions.append((lens_x, counter_y))
        ax.text(lens_x + 0.45, 0.9, 'Tangential arc', color='red',
                fontsize=annotation_font, ha='left')

        if show_rays:
            ax.annotate('', xy=arc_point, xytext=(source_x, source_y),
                        arrowprops=ray_style)
            ax.annotate('', xy=(obs_x, 0), xytext=arc_point,
                        arrowprops=ray_style)
            ax.annotate('', xy=(lens_x, counter_y), xytext=(source_x, source_y),
                        arrowprops=ray_style)
            ax.annotate('', xy=(obs_x, 0), xytext=(lens_x, counter_y),
                        arrowprops=ray_style)

    elif scenario == "quad":
        quad_angles = [35, 145, -35, -145]
        radii = [1.0, 0.95, 0.95, 1.0]
        for angle_deg, radius_scale in zip(quad_angles, radii):
            angle_rad = np.radians(angle_deg)
            x_img = lens_x + einstein_radius * radius_scale * np.cos(angle_rad)
            y_img = einstein_radius * radius_scale * np.sin(angle_rad)
            img = Circle((x_img, y_img), 0.045, color='red', alpha=0.6, zorder=4)
            ax.add_patch(img)
            image_positions.append((x_img, y_img))
        ax.text(lens_x + 0.35, 1.45, 'Four lensed images', color='red',
                fontsize=annotation_font, ha='left')

        if show_rays:
            for x_img, y_img in image_positions[::2]:
                ax.annotate('', xy=(x_img, y_img), xytext=(source_x, source_y),
                            arrowprops=ray_style)
                ax.annotate('', xy=(obs_x, 0), xytext=(x_img, y_img),
                            arrowprops=ray_style)

    else:
        image1_y = source_offset * 0.5 + einstein_radius * 0.9
        image2_y = source_offset * 0.5 - einstein_radius * 0.9

        image1 = Circle((lens_x, image1_y), 0.055,
                        color='red', alpha=0.6, zorder=4)
        image2 = Circle((lens_x, image2_y), 0.035,
                        color='red', alpha=0.4, zorder=4)
        ax.add_patch(image1)
        ax.add_patch(image2)
        image_positions.extend([(lens_x, image1_y), (lens_x, image2_y)])

        ax.text(lens_x + 0.3, image2_y - 0.2, 'Faint\ncounter-image',
                color='red', fontsize=annotation_font, ha='left', va='top')

        if show_rays:
            for y_pos in (image1_y, image2_y):
                ax.annotate('', xy=(lens_x, y_pos), xytext=(source_x, source_y),
                            arrowprops=ray_style)
                ax.annotate('', xy=(obs_x, 0), xytext=(lens_x, y_pos),
                            arrowprops=ray_style)

            mid_x1 = (source_x + lens_x) / 2
            mid_y1 = (source_y + image1_y) / 2
            ax.annotate('', xy=(mid_x1 + 0.2, mid_y1),
                        xytext=(mid_x1 - 0.2, mid_y1),
                        arrowprops=dict(arrowstyle='->', color='#FF8C42', linewidth=1.5))

    if show_angles:
        angle_radius = 0.45
        beta_angle = np.degrees(np.arctan2(source_y, source_x - obs_x))
        render_angle_indicator(
            ax,
            center=(obs_x, 0),
            radius=angle_radius,
            theta1=0,
            theta2=beta_angle,
            color='gray',
            label=r'$\beta$',
            label_offset=0.2,
        )

        if image_positions:
            image_angles = [
                np.degrees(np.arctan2(y_img, x_img - obs_x))
                for x_img, y_img in image_positions
            ]
            positive_angles = [a for a in image_angles if a >= 0]
            negative_angles = [a for a in image_angles if a < 0]

            if positive_angles:
                theta1_angle = max(positive_angles)
                render_angle_indicator(
                    ax,
                    center=(obs_x, 0),
                    radius=angle_radius + 0.18,
                    theta1=0,
                    theta2=theta1_angle,
                    color='black',
                    label=r'$\theta_1$',
                    label_offset=0.25,
                )

            if negative_angles:
                theta2_angle = min(negative_angles)
                render_angle_indicator(
                    ax,
                    center=(obs_x, 0),
                    radius=angle_radius + 0.33,
                    theta1=0,
                    theta2=theta2_angle,
                    color='black',
                    label=r'$\theta_2$',
                    label_offset=0.25,
                )

    if title:
        ax.text(0, -2.2, title, ha='center', fontsize=title_font, fontweight='bold')

    return ax

def save_figure(fig: plt.Figure, output_path: Path, dpi: int = 300) -> None:
    """Persist a Matplotlib figure to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')


def create_multi_panel_figure(
    configs: Iterable[SnapshotConfig] = None,
    *,
    output_dir: Path = Path("."),
    style: dict = DEFAULT_STYLE,
) -> plt.Figure:
    """
    Create a figure with multiple panels showing different configurations.
    """
    configs = tuple(configs) if configs is not None else default_snapshot_configs()
    output_dir = Path(output_dir)
    if len(configs) != 4:
        raise ValueError("Multi-panel figure requires exactly four snapshot configurations.")

    with plt.rc_context(style):
        fig = plt.figure(figsize=(14, 10), constrained_layout=True)
        gs = gridspec.GridSpec(2, 2, figure=fig)
        gs.update(hspace=0.35, wspace=0.25)

        axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
        for ax, config in zip(axes, configs):
            create_lensing_snapshot(
                ax,
                source_offset=config.source_offset,
                title=config.panel_title,
                scenario=config.scenario,
            )

        fig.suptitle(
            'Strong Gravitational Lensing: Source Position Effects',
            fontsize=plt.rcParams.get("axes.titlesize", 14) + 2,
            fontweight='bold',
            y=0.98,
        )
        save_figure(fig, output_dir / 'lensing_multi_panel.pdf')
        save_figure(fig, output_dir / 'lensing_multi_panel.png')
        print(f"Created: {output_dir / 'lensing_multi_panel.pdf'} and .png")

    return fig

def create_individual_snapshots(
    configs: Iterable[SnapshotConfig] = None,
    *,
    output_dir: Path = Path("."),
    style: dict = DEFAULT_STYLE,
) -> None:
    """
    Create individual figures for each configuration.
    """
    configs = tuple(configs) if configs is not None else default_snapshot_configs()
    output_dir = Path(output_dir)

    with plt.rc_context(style):
        for config in configs:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            create_lensing_snapshot(
                ax,
                source_offset=config.source_offset,
                title=config.individual_title,
                scenario=config.scenario,
            )

            pdf_path = output_dir / f"{config.filename_prefix}.pdf"
            png_path = output_dir / f"{config.filename_prefix}.png"
            save_figure(fig, pdf_path)
            save_figure(fig, png_path)
            plt.close(fig)
            print(f"Created: {pdf_path} and corresponding PNG")

def create_animation_frames(
    *,
    n_frames: int = 10,
    max_offset: float = 1.5,
    output_dir: Path = Path("."),
    style: dict = DEFAULT_STYLE,
) -> None:
    """
    Create frames that could be used for an animation or GIF.
    """
    output_dir = Path(output_dir)
    offsets = np.linspace(0, max_offset, n_frames)

    with plt.rc_context(style):
        for i, offset in enumerate(offsets):
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            create_lensing_snapshot(
                ax,
                source_offset=offset,
                title=f'β = {offset:.2f}',
                scenario="double",
            )

            frame_path = output_dir / f"frame_{i:03d}.png"
            save_figure(fig, frame_path, dpi=150)
            plt.close(fig)

    print(f"Created {n_frames} animation frames in {output_dir}")
    print("To create GIF, run: convert -delay 50 frame_*.png lensing_animation.gif")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate publication-quality lensing diagrams."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where figures will be written (default: current working directory).",
    )
    parser.add_argument(
        "--skip-multi",
        action="store_true",
        help="Skip generation of the multi-panel summary figure.",
    )
    parser.add_argument(
        "--skip-individual",
        action="store_true",
        help="Skip generation of individual snapshot figures.",
    )
    parser.add_argument(
        "--make-frames",
        action="store_true",
        help="Create animation frames across a range of source offsets.",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=10,
        help="Number of animation frames to emit when --make-frames is used.",
    )
    parser.add_argument(
        "--max-offset",
        type=float,
        default=1.5,
        help="Maximum source offset (in Einstein radii) for animation frames.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    configs = default_snapshot_configs()

    print("Generating static versions of interactive lensing diagrams...")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    if not args.skip_multi:
        multi_fig = create_multi_panel_figure(configs, output_dir=output_dir)
        plt.close(multi_fig)

    if not args.skip_individual:
        create_individual_snapshots(configs, output_dir=output_dir)

    if args.make_frames:
        create_animation_frames(
            n_frames=args.n_frames,
            max_offset=args.max_offset,
            output_dir=output_dir,
        )

    print("=" * 60)
    print("All requested figures generated successfully!")
    print("Refer to interactive_content_guide.md for integration guidance.")


if __name__ == "__main__":
    main()
