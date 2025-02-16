"""
IMU Data Annotation and Visualization Tool with Outlier Detection

Features:
1. Browse and delete CSV data files
2. 3D pose visualization
3. Multi-dimensional sensor data visualization
4. Adjustable playback parameters
5. Box plot visualization
6. Z-score based outlier detection and cleaning
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
from scipy import stats
import ttkbootstrap as tb
from ttkbootstrap.constants import *

# === Constant Configuration ===
CONFIG = {
    "SAVE_PATH": "./Dataset/",
    "EXPECTED_COLUMNS": ['Pitch', 'Roll', 'Gyro_X', 'Gyro_Y', 'Gyro_Z',
                         'Accel_X', 'Accel_Y', 'Accel_Z'],
    "INITIAL_GEOMETRY": "1850x700+0+0",
    "PLOT_WINDOW_GEOMETRY": "1920x780+0+300",
    "CUBE_SCALE": 0.5,
    "DEFAULT_SKIP": 20,
    "DEFAULT_SPEED": 10,
    "ZSCORE_THRESHOLD": 3,
    "OUTLIER_COLOR": "red",
    "CLEANED_SUFFIX": "_cleaned"
}


class IMUVisualizer:
    """Main controller for IMU data visualization application with outlier handling"""

    def __init__(self):
        self.csv_files = []
        self.current_file_index = 0
        self.plot_window = None
        self.ani = None

        # Initialize main application window
        self.app = tb.Window(themename="journal")
        self._setup_gui()
        self._refresh_file_list()

    def _setup_gui(self):
        """Initialize graphical user interface components"""
        self.app.title("IMU Data Annotation Tool")
        self.app.geometry(CONFIG["INITIAL_GEOMETRY"])

        # Main container frame
        main_frame = ttk.Frame(self.app, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # File listbox configuration
        self.file_listbox = tk.Listbox(main_frame, height=10, width=60)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.file_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        # Control panel configuration
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create interface components
        self._create_button_panel(control_frame)
        self._create_slider_panel(control_frame)

    def _create_button_panel(self, parent):
        """Create button control panel with navigation and actions"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(pady=10)

        buttons = [
            ("⬅ Prev", self._prev_file, PRIMARY),
            ("Next ➡", self._next_file, PRIMARY),
            ("🗑 Delete", self._delete_file, DANGER),
            ("📊 Visualize", self._visualize_selected_file, SUCCESS),
            ("📦 Box Plot", self._show_boxplot, INFO),
            ("🧹 Clean Data", self._clean_outliers, WARNING)
        ]

        for col, (text, command, style) in enumerate(buttons):
            btn = ttk.Button(
                button_frame,
                text=text,
                width=10,
                command=command,
                bootstyle=style
            )
            btn.grid(row=0, column=col, padx=5, pady=5)

    def _create_slider_panel(self, parent):
        """Create panel with playback control sliders"""
        slider_frame = ttk.Frame(parent)
        slider_frame.pack(pady=20, fill=tk.X)

        # Playback speed control slider
        self.speed_slider = self._create_slider(
            slider_frame,
            "Playback Speed (ms)",
            10, 200,
            CONFIG["DEFAULT_SPEED"],
            self._update_speed_label
        )

        # Frame skip control slider
        self.skip_slider = self._create_slider(
            slider_frame,
            "Frame Skip Ratio",
            1, 50,
            CONFIG["DEFAULT_SKIP"],
            self._update_skip_label
        )

    def _create_slider(self, parent, label, from_, to, default, callback):
        """Generic slider component creation"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)

        ttk.Label(frame, text=label).pack(anchor=tk.W)

        slider = tk.Scale(
            frame,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            command=callback,
            length=300
        )
        slider.set(default)
        slider.pack(fill=tk.X)

        value_label = ttk.Label(frame, text=f"{label.split('(')[0]}: {default}")
        value_label.pack(anchor=tk.W)

        return slider

    def _refresh_file_list(self):
        """Refresh list of available CSV files"""
        self.csv_files = [
            f for f in os.listdir(CONFIG["SAVE_PATH"])
            if f.endswith('.csv')
        ]
        self.file_listbox.delete(0, tk.END)
        for file in self.csv_files:
            self.file_listbox.insert(tk.END, file)

    def _on_listbox_select(self, event):
        """Handle file selection event in listbox"""
        selection = event.widget.curselection()
        if selection:
            self._update_file_selection(selection[0])

    def _update_file_selection(self, index):
        """Update currently selected file index"""
        if 0 <= index < len(self.csv_files):
            self.current_file_index = index
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(index)
            self.file_listbox.see(index)

    def _delete_file(self):
        """Delete currently selected CSV file"""
        if not self.csv_files:
            return

        confirm = messagebox.askyesno(
            "Confirm Delete",
            f"Delete {self.csv_files[self.current_file_index]}?",
            parent=self.app
        )

        if confirm:
            file_path = os.path.join(
                CONFIG["SAVE_PATH"],
                self.csv_files[self.current_file_index]
            )
            os.remove(file_path)
            self._refresh_file_list()
            self.current_file_index = max(
                0,
                min(self.current_file_index, len(self.csv_files) - 1)
            )
            self._close_plot_window()

    def _visualize_selected_file(self):
        """Launch visualization for selected data file"""
        self._close_plot_window()

        if not self.csv_files:
            messagebox.showwarning("No Data", "No CSV files available!")
            return

        try:
            df = self._load_and_validate_data()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        # Create visualization window
        self.plot_window = tk.Toplevel()
        self.plot_window.title(
            f"Visualization - {self.csv_files[self.current_file_index]}"
        )

        # Initialize visualization components
        fig, (ax3d, ax_pitch, ax_gyro, ax_accel) = self._create_figure()
        cube = self._init_3d_cube(ax3d)
        self._plot_2d_data(ax_pitch, ax_gyro, ax_accel, df)

        # Embed matplotlib canvas
        canvas = FigureCanvasTkAgg(fig, master=self.plot_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Configure animation
        self._setup_animation(canvas, fig, df, cube, ax3d, ax_pitch, ax_gyro, ax_accel)

    def _load_and_validate_data(self):
        """Load and validate CSV data file"""
        file_path = os.path.join(
            CONFIG["SAVE_PATH"],
            self.csv_files[self.current_file_index]
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)

        missing_cols = set(CONFIG["EXPECTED_COLUMNS"]) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing columns in {os.path.basename(file_path)}: {', '.join(missing_cols)}"
            )

        # Apply frame skipping
        frame_skip = int(self.skip_slider.get())
        return df.iloc[::frame_skip].reset_index(drop=True)

    def _create_figure(self):
        """Create matplotlib figure layout"""
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig,
                      width_ratios=[1, 1],
                      height_ratios=[1, 1],
                      wspace=0.1, hspace=0.2)

        ax3d = fig.add_subplot(gs[0, 0], projection='3d')
        ax_pitch = fig.add_subplot(gs[0, 1])
        ax_gyro = fig.add_subplot(gs[1, 0])
        ax_accel = fig.add_subplot(gs[1, 1])

        return fig, (ax3d, ax_pitch, ax_gyro, ax_accel)

    def _init_3d_cube(self, ax):
        """Initialize 3D cube visualization"""
        cube_faces = self._create_cube()
        cube = Poly3DCollection(cube_faces, alpha=0.5, edgecolor='k')
        ax.add_collection3d(cube)
        ax.set(
            xlim=[-1, 1],
            ylim=[-1, 1],
            zlim=[-1, 1],
            xlabel='X',
            ylabel='Y',
            zlabel='Z'
        )
        return cube_faces

    def _plot_2d_data(self, ax_pitch, ax_gyro, ax_accel, df):
        """Plot 2D sensor data charts with outlier highlighting"""
        time_steps = np.arange(len(df))

        # Calculate outliers
        z_scores = np.abs(stats.zscore(df[CONFIG["EXPECTED_COLUMNS"]]))
        outliers = (z_scores > CONFIG["ZSCORE_THRESHOLD"]).any(axis=1)

        # Orientation plot
        ax_pitch.plot(time_steps, df['Pitch'].fillna(0), 'r', label='Pitch')
        ax_pitch.plot(time_steps, df['Roll'].fillna(0), 'g', label='Roll')
        ax_pitch.scatter(time_steps[outliers], df['Pitch'][outliers],
                         color=CONFIG["OUTLIER_COLOR"], zorder=3)
        ax_pitch.scatter(time_steps[outliers], df['Roll'][outliers],
                         color=CONFIG["OUTLIER_COLOR"], zorder=3)
        ax_pitch.set_title("Orientation Data")
        ax_pitch.legend()
        ax_pitch.grid(True)

        # Gyroscope plot
        for col, color in zip(['Gyro_X', 'Gyro_Y', 'Gyro_Z'], ['b', 'g', 'r']):
            ax_gyro.plot(time_steps, df[col], color, label=col)
            ax_gyro.scatter(time_steps[outliers], df[col][outliers],
                            color=CONFIG["OUTLIER_COLOR"], zorder=3)
        ax_gyro.set_title("Gyroscope")
        ax_gyro.legend()
        ax_gyro.grid(True)

        # Accelerometer plot
        for col, color in zip(['Accel_X', 'Accel_Y', 'Accel_Z'], ['b', 'g', 'r']):
            ax_accel.plot(time_steps, df[col], color, label=col)
            ax_accel.scatter(time_steps[outliers], df[col][outliers],
                             color=CONFIG["OUTLIER_COLOR"], zorder=3)
        ax_accel.set_title("Accelerometer")
        ax_accel.legend()
        ax_accel.grid(True)

    def _setup_animation(self, canvas, fig, df, cube_faces, ax3d, ax_pitch, ax_gyro, ax_accel):
        """Configure animation parameters and callbacks"""
        pitch = df['Pitch'].fillna(0)
        roll = df['Roll'].fillna(0)
        play_speed = int(self.speed_slider.get())

        def update_frame(i):
            """Animation frame update callback"""
            if i >= len(df):
                return

            # Update 3D cube rotation
            rot = R.from_euler('yx', [pitch[i], roll[i]]).as_matrix()
            rotated = [[rot @ vertex for vertex in face] for face in cube_faces]
            ax3d.collections[0].set_verts(rotated)
            ax3d.set_title(f"Frame {i + 1}/{len(df)}")

            # Update vertical cursors in 2D plots
            for ax in [ax_pitch, ax_gyro, ax_accel]:
                for line in ax.lines:
                    if line.get_label() == '_cursor':
                        line.remove()
                ax.axvline(x=i, color='k', linestyle='--', alpha=0.5, label='_cursor')

            canvas.draw_idle()

        # Initialize animation
        self.ani = animation.FuncAnimation(
            fig,
            update_frame,
            frames=len(df),
            interval=play_speed,
            repeat=False
        )

        # Window close handler
        def on_closing():
            if self.ani and self.ani.event_source:
                self.ani.event_source.stop()
            self.plot_window.destroy()
            self.plot_window = None

        self.plot_window.protocol("WM_DELETE_WINDOW", on_closing)
        self._adjust_window_geometry()

    def _adjust_window_geometry(self):
        """Adjust visualization window geometry"""

        def force_redraw():
            self.plot_window.update_idletasks()
            self.plot_window.geometry(CONFIG["PLOT_WINDOW_GEOMETRY"])
            self.plot_window.update()

        self.plot_window.geometry("0x0+0+0")
        self.plot_window.after(100, force_redraw)

    def _show_boxplot(self):
        """Display boxplot for outlier detection using Tkinter canvas"""
        if not self.csv_files:
            messagebox.showwarning("No Data", "No CSV files available!")
            return

        try:
            df = self._load_and_validate_data()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        # Create new window
        boxplot_window = tk.Toplevel(self.app)
        boxplot_window.title("Box Plot Analysis")

        # Create figure
        fig = plt.Figure(figsize=(10, 8))
        axes = fig.subplots(3, 1)

        features = ['Gyro_X', 'Gyro_Y', 'Gyro_Z',
                    'Accel_X', 'Accel_Y', 'Accel_Z']

        # Create boxplots
        for i, col in enumerate(features):
            ax = axes[i // 2]
            ax.boxplot(df[col], positions=[i % 2],
                       widths=0.6, patch_artist=True)
            ax.set_title(f'Box Plot - {col.split("_")[0]}')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(features[i // 2 * 2:i // 2 * 2 + 2])

        fig.tight_layout()

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=boxplot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add close button
        ttk.Button(
            boxplot_window,
            text="Close",
            command=boxplot_window.destroy
        ).pack(pady=10)

    def _clean_outliers(self):
        """Clean data using Z-score outlier detection"""
        if not self.csv_files:
            messagebox.showwarning("No Data", "No CSV files available!")
            return

        try:
            df = self._load_and_validate_data()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(df[CONFIG["EXPECTED_COLUMNS"]]))
        outlier_mask = (z_scores > CONFIG["ZSCORE_THRESHOLD"]).any(axis=1)

        # Create cleaned dataframe
        cleaned_df = df[~outlier_mask].copy()
        original_count = len(df)
        cleaned_count = len(cleaned_df)
        removed_count = original_count - cleaned_count

        # Show confirmation dialog
        confirm = messagebox.askyesno(
            "Confirm Cleaning",
            f"Found {removed_count} outliers ({removed_count / original_count:.1%})\n"
            f"Save cleaned data?",
            parent=self.app
        )

        if confirm:
            # Save cleaned data
            original_path = os.path.join(
                CONFIG["SAVE_PATH"],
                self.csv_files[self.current_file_index]
            )
            new_filename = os.path.splitext(self.csv_files[self.current_file_index])[0] + \
                           CONFIG["CLEANED_SUFFIX"] + ".csv"
            new_path = os.path.join(CONFIG["SAVE_PATH"], new_filename)
            cleaned_df.to_csv(new_path, index=False)
            messagebox.showinfo("Success",
                                f"Saved cleaned data to {new_filename}")
            self._refresh_file_list()

    @staticmethod
    def _create_cube():
        """Generate cube vertices for 3D visualization"""
        scale = CONFIG["CUBE_SCALE"]
        vertices = np.array([
            [-scale, -scale, -scale],
            [scale, -scale, -scale],
            [scale, scale, -scale],
            [-scale, scale, -scale],
            [-scale, -scale, scale],
            [scale, -scale, scale],
            [scale, scale, scale],
            [-scale, scale, scale]
        ])
        return [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[3], vertices[7], vertices[4]]
        ]

    def _close_plot_window(self):
        """Close visualization window and cleanup resources"""
        if self.plot_window and self.plot_window and self.plot_window.destroy:
            self.plot_window.destroy()
            self.plot_window = None
        if self.ani and self.ani.event_source and self.ani.event_source.stop:
            self.ani.event_source.stop()
            self.ani = None

    def _prev_file(self):
        """Navigate to previous file in list"""
        if self.current_file_index > 0:
            self._close_plot_window()
            self._update_file_selection(self.current_file_index - 1)
            self._visualize_selected_file()

    def _next_file(self):
        """Navigate to next file in list"""
        if self.current_file_index < len(self.csv_files) - 1:
            self._close_plot_window()
            self._update_file_selection(self.current_file_index + 1)
            self._visualize_selected_file()

    def _update_speed_label(self, value):
        """Update playback speed label text"""
        self.speed_slider.master.children['!label'].config(
            text=f"Speed: {int(float(value))} ms"
        )

    def _update_skip_label(self, value):
        """Update frame skip label text"""
        self.skip_slider.master.children['!label'].config(
            text=f"Skip: {int(float(value))}"
        )

    def run(self):
        """Start main application loop"""
        self.app.mainloop()


if __name__ == "__main__":
    visualizer = IMUVisualizer()
    visualizer.run()
