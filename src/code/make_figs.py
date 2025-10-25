from turtle import back
from xml.dom.minidom import Element
from .analysis_utils import *
from dotenv import load_dotenv
from pyprojroot import here
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_hex, ListedColormap
import matplotlib.font_manager as fm
from matplotlib.ticker import ScalarFormatter, FixedLocator
from matplotlib.collections import LineCollection
from scipy.interpolate import interp1d
from tqdm import tqdm
import math
from scipy.optimize import curve_fit
from reportlab.pdfgen import canvas
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.utils import ImageReader
import tempfile
import subprocess
from pypdf import PdfReader, PdfWriter, PageObject, Transformation
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


class FigureGenerator:
    def __init__(self, transformer_df, transformer_df_all_checkpoints, algo_df, setting, figs_dir=here("figures"), include_optimal_constant_solution=False):
        self.transformer_df_all_checkpoints = transformer_df_all_checkpoints
        self.transformer_df = transformer_df
        self.algo_df = algo_df
        self.setting = setting
        self.figs_dir = figs_dir
        self.set_plotting_constants()
        self.algo_names_dict = self.algo_df.iloc[0]['config'].algo_names_dict
        if include_optimal_constant_solution:
            self.algo_names_dict["optimal_constant"] = "$C$"
        self.algo_colors_dict = self.map_algo_names_to_colors(self.algo_names_dict)

        os.makedirs(self.figs_dir, exist_ok=True)

        # Dictionary mapping markdown description titles to figure generation functions
        self.figure_generators = {
            "transience": self.generate_transience_plot,
            "transience appendix": self.generate_transience_plot,
            "task diversity threshold": self.generate_task_diversity_threshold_plot, 
            "task diversity threshold appendix": self.generate_task_diversity_threshold_plot,
            "relative distance": self.generate_relative_distance_plot, 
            "prediction comparison": self.generate_prediction_comparison_plot, 
            "posterior odds": self.generate_posterior_odds_plot, 
            "algorithm probabilities": self.generate_algorithm_probabilities_plot, 
            "sublinear evidence accumulation": self.generate_sublinear_evidence_accumulation_plot, 
            "vector field": self.generate_vector_field_plot, 
            "transience predictions": self.generate_transience_predictions_plot, 
            "titration": self.generate_titration_plot,
            "free parameter titration": self.generate_free_parameter_titration_plot,
            "relative and absolute distance": self.generate_relative_and_absolute_distance_plot,
            "loss titration": self.generate_loss_titration_plot,
            "interpolation threshold": self.generate_interpolation_threshold_plot,
        }

    def set_plotting_constants(self):
        # set var to title
        self.var_to_title = {
            "num_dims": "Task Dimensionality",
            "num_tasks": "Task Diversity $D$",
            "context_length": "Context Length",
            "mlp_expansion_factor": "MLP Width",
            "checkpoint": "Training Steps $N$",
            "beta": "Complexity Penalty $\\beta$",
            "gamma": "Evidence Accumulation Rate $\\gamma$",
            "alpha": "Likelihood Exponent $1-\\alpha$",
        }

        # define colors
        self.offwhite = "#F9F9F9"
        self.blue = "#699AEF"
        self.red = "#DF4A44"
        self.orange = "#F19E4B"
        self.green = "#74D34D"
        self.teal = "#44DFD7"
        
        # define colormaps
        self.blue_to_white = LinearSegmentedColormap.from_list('', [self.blue, 'white'])
        self.white_to_red = LinearSegmentedColormap.from_list('', ['white', self.red])
        self.blue_to_white_to_red = LinearSegmentedColormap.from_list('', [self.blue, 'white', self.red])
        self.blue_to_offwhite_to_red = LinearSegmentedColormap.from_list('', [self.blue, self.offwhite, self.red])
        
        # define font sizes
        self.title_fontsize = 30
        self.axis_label_fontsize = 30
        self.tick_fontsize = 22
        self.legend_fontsize = 20
        self.sideplot_title_fontsize = 25
        self.sideplot_axis_label_fontsize = 16
        self.sideplot_tick_fontsize = 14
        self.sideplot_legend_fontsize = 15
        self.sideplot_colorbar_fontsize = 15

        self.arrow_width = 0.08  # Width of arrow head relative to figure
        self.arrow_head_length = 0.12  # Length of arrow head relative to figure
        self.arrow_head_width = 0.10  # Width of arrow head at its base relative to figure
        self.arrow_color = 'black'
        self.arrow_linewidth = 2.5

        # set matplotlib to not display right and top spines by default
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False

        # Find and register Avenir font
        avenir_fonts = [font for font in fm.findSystemFonts(fontext='ttf') if 'Avenir' in font]
        if avenir_fonts:
            avenir_path = avenir_fonts[0]
            # Register the font with matplotlib
            font_prop = fm.FontProperties(fname=avenir_path)
            font_name = font_prop.get_name()
            fm.fontManager.addfont(avenir_path)
            plt.rcParams['font.family'] = font_name
            plt.rcParams["axes.labelweight"] = "light"
            print(f"Registered Avenir font from: {avenir_path}")
        else:
            plt.rcParams['font.family'] = 'serif'
            print("Avenir font not found, using serif instead.")

        plt.rcParams['mathtext.fontset'] = 'cm'  # 'cm' stands for Computer Modern (standard LaTeX font)

        # Ensure text remains as vector text, not outlines
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42

    def map_algo_names_to_colors(self, algo_names_dict):
        algo_colors_dict = {}
        for algo_name in algo_names_dict:
            if algo_name == "memorized":
                algo_colors_dict[algo_name] = self.red
            elif algo_name == "generalized":
                algo_colors_dict[algo_name] = self.blue
            elif algo_name == "optimal_constant":
                algo_colors_dict[algo_name] = self.teal
        return algo_colors_dict

    # define weighted color combination function
    def weighted_color_combination(self, weights):
        """
        Create a color that represents a weighted combination of algorithm colors.
        
        Args:
            weights (list or np.array): Weight vector where each element corresponds to 
                                        the weight for the corresponding algorithm color in self.algo_colors_dict
                                        
        Returns:
            str: Hex color string representing the weighted combination
        """
        # Convert weights to numpy array and normalize
        weights = np.array(weights)
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
                
        # Convert hex colors to RGB
        rgb_colors = [to_rgb(color) for color in self.algo_colors_dict.values()]
        
        # Calculate weighted average in RGB space
        weighted_rgb = np.zeros(3)
        for i, (weight, rgb) in enumerate(zip(weights, rgb_colors)):
            weighted_rgb += weight * np.array(rgb)
        
        # Ensure RGB values are in valid range [0, 1]
        weighted_rgb = np.clip(weighted_rgb, 0, 1)
        
        # Convert back to hex
        return to_hex(weighted_rgb)

    def convert_barycentric_weights_to_colormap(self, df, col_name):
        """
        Convert barycentric weights to colors and create a discrete colormap.
        
        Args:
            df: DataFrame containing the barycentric weights
            col_name: Name of the column containing barycentric weights
            
        Returns:
            tuple: (plot_df, plot_column, plot_cmap, vmin, vmax)
        """
        # Create a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()
        
        # Create a mapping from each unique color to a numeric index
        color_to_index = {}
        index_to_color = {}
        next_index = 0
        numeric_values = []
        
        for idx, row in df_copy.iterrows():
            weights = row[col_name]
            # Use the weighted color combination function
            color = self.weighted_color_combination(weights)
            
            # Map this color to a numeric index
            if color not in color_to_index:
                color_to_index[color] = next_index
                index_to_color[next_index] = color
                next_index += 1
            
            numeric_values.append(color_to_index[color])
        
        # Add the numeric column to the dataframe
        df_copy[f"{col_name}_numeric"] = numeric_values
                    
        # Create ordered list of colors based on their indices
        ordered_colors = [index_to_color[i] for i in range(next_index)]
        
        # Ensure we have at least one color for the colormap
        vmin, vmax = 0, next_index - 1
        custom_cmap = ListedColormap(ordered_colors)
        
        # Use the numeric column and custom colormap
        plot_column = f"{col_name}_numeric"
        plot_df = df_copy
        plot_cmap = custom_cmap
        
        return plot_df, plot_column, plot_cmap, vmin, vmax

    def get_metric_name(self, distance = False):
        if self.setting == "categorical-sequence":
            return "KL"
        elif self.setting == "linear-regression":
            return "MSE"
        elif self.setting == "classification":
            if distance:
                return "KL"
            else:
                return "Cross-entropy" 

    
    def format_plot_name(self, setting, fixed_values, plot_name, extension="pdf"):
        """
        Format the plot name based on the setting and fixed values.
        
        Args:
            setting (str): The setting name
            fixed_values (dict): A dictionary of fixed values
            
        Returns:
            str: The formatted plot name
        """
        os.makedirs(os.path.join(self.figs_dir, setting, plot_name), exist_ok=True)
        return os.path.join(self.figs_dir, setting, plot_name, f"{plot_name}-{setting}-{'__'.join(f'{v}{k}' for k, v in fixed_values.items())}.{extension}")
        
    @staticmethod
    def save_figure(fig, filepath, format='pdf', dpi=1000):
        """Save the figure to disk with the given name in publication-quality format"""        
        # Selectively rasterize complex elements while keeping axes, text, etc. as vector
        for ax in fig.get_axes():
            # Rasterize heatmaps and image plots
            for collection in ax.collections:
                if collection.__class__.__name__ in ['QuadMesh', 'AxesImage']:
                    collection.set_rasterized(True)
            
            # Rasterize very dense scatter plots (>1000 points)
            for collection in ax.collections:
                if hasattr(collection, '_offsets') and hasattr(collection._offsets, 'shape'):
                    if collection._offsets.shape[0] > 500:
                        collection.set_rasterized(True)
                        print(f"Rasterized {collection._offsets.shape[0]} points")
            
        # Save the figure
        fig.savefig(filepath, bbox_inches='tight', dpi=dpi, format=format)
        print(f"Saved figure to {filepath}")

    def make_figs(self, fig_configs):
        """
        Generate figures used in the paper.
        
        Args:
            fig_configs (dict): Dictionary mapping figure names to configurations with parameters needed for that figure
        """
        # Generate the requested figures with provided configurations
        for fig_name, config in tqdm(fig_configs.items()):
            if fig_name in self.figure_generators:
                print(f"Generating figure: {fig_name}")
                if fig_name == "transience appendix" or fig_name == "task diversity threshold appendix":
                    config["main"] = False
                self.figure_generators[fig_name](config, show=False, save=True)
            else:
                print(f"No generator found for '{fig_name}'") 

    def generate_transience_plot(self, config, show=False, save=True):
        """
        Generate a plot showing transience in the model's performance.
        
        Args:
            config (dict): A dictionary of configuration parameters, with the following keys:
                fixed_values: {context_length, mlp_expansion_factor, num_dims}
                mode: The evaluation mode (eval, val, train)
                x_scale: The x-axis scale (linear, log)
                main: Whether this is the main plot
                custom_name: The custom name of the plot (if not provided, the default name will be used)
            show (bool): Whether to display the plot
            save (bool): Whether to save the plot
        """
        fixed_values = config["fixed_values"]
        mode = config["mode"]
        x_scale = config["x_scale"]
        main = config.get("main", True)
        
        if main:
            fig, ax = plot_flexible_scatterplot(
                df=self.transformer_df_all_checkpoints,
                variable_config={
                    "fixed": ["context_length", "mlp_expansion_factor", "num_dims", "num_tasks"],
                    "iterate_over": "checkpoint",
                    "grid_by": None,
                },
                fixed_values=fixed_values,
                value_column=[lambda x: x[f"{mode}_metrics"].get(f'{mode}_metric', float('nan'))]+\
                            [lambda x, algo_name=algo_name: self.algo_df.query("num_tasks == @x.num_tasks & num_dims == @x.num_dims & context_length == @x.context_length", engine="python").iloc[0][f"{algo_name}_{mode}_metrics"].get(f'{mode}_metric', float('nan')) for algo_name in self.algo_names_dict],
                needed_cols_for_callable=[[f"{mode}_metrics"]],
                x_scale=x_scale,
                x_scale_base=10,
                y_label=self.get_metric_name(),
                color=["black"] + [self.algo_colors_dict[algo_name] for algo_name in self.algo_names_dict],
                linewidth=7,
                linestyle=["-", "--", "--"],
                marker=["", "", ""],
                legend_labels=["Transformer"] + [self.algo_names_dict[algo_name] for algo_name in self.algo_names_dict],
                legend_loc="upper right",
                show_title=False,
                axis_label_fontsize=self.axis_label_fontsize,
                tick_fontsize=self.tick_fontsize,
                legend_fontsize=self.legend_fontsize,
                figsize=(10, 6)
            )
            # drop legend
            ax.legend_.remove()

            # drop box lines, axis labels, and ticks
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False, bottom=False, left=False)
        else:
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get unique num_tasks values and sort them
            filtered_df = self.transformer_df_all_checkpoints.query(
                "context_length == @fixed_values['context_length'] and "
                "mlp_expansion_factor == @fixed_values['mlp_expansion_factor'] and "
                "num_dims == @fixed_values['num_dims']", engine="python"
            )
            num_tasks_values = sorted(filtered_df['num_tasks'].unique())
            
            # Find the min and max values for normalization
            min_val = filtered_df[f"{mode}_metrics"].apply(lambda x: x.get(f'{mode}_metric', float('nan'))).min()
            max_val = filtered_df[f"{mode}_metrics"].apply(lambda x: x.get(f'{mode}_metric', float('nan'))).max()
            
            # Choose which tasks to label - you can adjust this as needed
            tasks_to_label = config["tasks_to_label"] if config.get("tasks_to_label", None) is not None else [4, 16, 128, 4096]
            
            # Plot each task diversity as a separate line with black-to-grey gradient
            for i, num_tasks in enumerate(num_tasks_values):
                task_data = filtered_df[filtered_df['num_tasks'] == num_tasks].sort_values('checkpoint')
                
                if len(task_data) < 3:  # Skip if not enough data points
                    continue
                
                x_data = task_data['checkpoint'].values
                y_data = task_data[f"{mode}_metrics"].apply(lambda x: x.get(f'{mode}_metric', float('nan'))).values
                
                # Skip if we have NaN values
                if np.isnan(y_data).any():
                    continue
                    
                # Calculate shade of grey based on position in the list
                # Smaller D values get darker colors (closer to black)
                darkness = 1 - (i / 12)
                color = plt.cm.Greys(darkness)
                
                # Plot a single colored line based on task diversity
                ax.plot(x_data, y_data, color=color, linewidth=2)
                
                # Add task diversity label at the end of the curve for selected tasks
                if num_tasks in tasks_to_label and len(x_data) > 0:
                    ax.text(
                        x_data[-1] * 1.02,
                        y_data[-1],
                        f"$D$={num_tasks}",
                        fontsize=self.sideplot_legend_fontsize,
                        ha='left',
                        va='center',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                    )
                
                # Plot the generalized solution ($G$) with dashed line
                try:
                    # Get the generalized solution value for this task diversity
                    g_value = self.algo_df.query(
                        "num_tasks == @num_tasks & num_dims == @fixed_values['num_dims'] & "
                        "context_length == @fixed_values['context_length']", engine="python"
                    ).iloc[0][f"generalized_{mode}_metrics"].get(f'{mode}_metric', float('nan'))
                    
                    # Only plot if we have a valid value
                    if not np.isnan(g_value):
                        ax.plot(x_data, [g_value] * len(x_data), 
                                color=self.blue, linestyle='--', linewidth=2, alpha=0.6)
                        
                        # Add G label at the beginning of each generalized solution line
                        if num_tasks in tasks_to_label:
                            ax.text(
                                x_data[0] * 0.93,  # Slightly left of the first point
                                g_value, 
                                "$G$",
                                fontsize=self.sideplot_legend_fontsize,
                                ha='right',
                                va='center',
                                color=self.blue,
                                bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1)
                            )
                except (IndexError, KeyError):
                    # Skip if we can't find the generalized solution
                    pass
            
            # Set the limits correctly
            if 'x_data' in locals():  # Check if x_data exists (at least one curve was plotted)
                ax.set_xlim(x_data.min(), x_data.max() * 1.05)
                ax.set_ylim(min(min_val * 0.95, ax.get_ylim()[0]), max(max_val * 1.05, ax.get_ylim()[1]))
            
            # Configure axes
            ax.set_xlabel(self.var_to_title["checkpoint"], fontsize=self.axis_label_fontsize)
            ax.set_ylabel(self.get_metric_name(), fontsize=self.axis_label_fontsize)
            ax.tick_params(axis='both', labelsize=self.tick_fontsize)
            
            # Set x-axis scale
            ax.set_xscale(x_scale)
            if x_scale == 'linear':
                format_axis_labels_in_thousands(ax, axis='x')
            
        if save:
            if config.get("custom_name", None) is not None:
                filepath = self.format_plot_name(self.setting, fixed_values, config["custom_name"])
            else:
                filepath = self.format_plot_name(self.setting, fixed_values, f"transience_{mode}{'_appendix' if not main else ''}")
            self.save_figure(fig, filepath)

        if show:
            plt.show()
            plt.close(fig)

    def generate_task_diversity_threshold_plot(self, config, show=False, save=True):
        """
        Generate the task diversity threshold plot.
        
        Args:
            config (dict): A dictionary of configuration parameters, with the following keys:
                fixed_values: {context_length, mlp_expansion_factor, num_dims, checkpoint}
                main: Whether to plot the main paper plot
                y_axis_limits: The y-axis limits to use for the plot
                custom_name: The custom name of the plot (if not provided, the default name will be used)
            show (bool): Whether to display the plot
            save (bool): Whether to save the plot
        """
        main = config["main"] if config.get("main", None) is not None else True
        if main:
            line_width = 7
        else:
            line_width = 3

        train_mode = "train" if config.get("train_mode", None) is None else config["train_mode"]

        for mode in [train_mode, "eval"]:

            fig, ax = plot_flexible_scatterplot(
                df=self.transformer_df,
                variable_config={
                    "fixed": ["context_length", "mlp_expansion_factor", "num_dims", "checkpoint"],
                    "iterate_over": "num_tasks",
                    "grid_by": None,
                },
                fixed_values=config["fixed_values"],
                value_column=[lambda x: x[f"{mode}_metrics"].get(f'{mode}_metric', float('nan'))]+\
                             [lambda x, algo_name=algo_name: self.algo_df.query("num_tasks == @x.num_tasks & num_dims == @x.num_dims & context_length == @x.context_length", engine="python").iloc[0][f"{algo_name}_{mode}_metrics"].get(f'{mode}_metric', float('nan')) for algo_name in self.algo_names_dict],
                needed_cols_for_callable=[[f"{mode}_metrics"]],
                x_scale="log",
                x_scale_base=2,
                linewidth=line_width,
                color=["black"] + [self.algo_colors_dict[algo_name] for algo_name in self.algo_names_dict],
                linestyle=["-", *["--" for _ in range(len(self.algo_names_dict))]],
                marker=["", *["" for _ in range(len(self.algo_names_dict))]],
                legend_labels=["Transformer"] + [self.algo_names_dict[algo_name] for algo_name in self.algo_names_dict],
                y_label = self.get_metric_name(),
                x_label = self.var_to_title["num_tasks"],
                legend_loc="lower right",
                show_title=False,
                axis_label_fontsize=self.axis_label_fontsize,
                tick_fontsize=self.tick_fontsize,
                legend_fontsize=self.legend_fontsize,
                figsize=(10, 6)
            )
            # drop legend
            ax.legend_.remove()


            if main:
                # drop box lines, axis labels, and ticks
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False, bottom=False, left=False)

            if save:
                main_suffix = "" if main else "_appendix"
                if config.get("custom_name", None) is not None:
                    filepath = self.format_plot_name(self.setting, config["fixed_values"], f"{config['custom_name']}{main_suffix}")
                else:
                    filepath = self.format_plot_name(self.setting, config["fixed_values"], f"task_diversity_threshold_{mode}{main_suffix}")
                self.save_figure(fig, filepath)

            if show:
                plt.show()
                plt.close(fig)

    def generate_relative_distance_plot(self, config, show=False, save=True):
        """
        Generate the relative distance plot.

        Args:
            config (dict): A dictionary of configuration parameters, with the following keys:
                fixed_values: {mlp_expansion_factor, context_length, num_dims}
                x_annotated: The x-axis index to annotate
                y_annotated: The y-axis index to annotate
                col_name [optional]: The column name to plot
                mode [optional]: The mode to plot
                custom_name: The custom name of the plot (if not provided, the default name will be used)
            show (bool): Whether to display the plot
            save (bool): Whether to save the plot
        """
        mode = "train" if config.get("mode", None) is None else config["mode"]
        if config.get("col_name", None) is None:
            if len(self.algo_names_dict) == 2:
                col_name = f"relative_distance_{mode}"
            else:
                col_name = f"barycentric_weights_{mode}"
        else:
            col_name = config["col_name"]

        # Handle barycentric weights by preprocessing them into colors
        # When there are >2 algorithms, barycentric_weights contains numpy arrays of weights
        # We convert these to weighted color combinations and create a discrete colormap
        use_barycentric_colors = (len(self.algo_names_dict) > 2 and col_name == f"barycentric_weights_{mode}")
        if use_barycentric_colors:
            plot_df, plot_column, plot_cmap, vmin, vmax = self.convert_barycentric_weights_to_colormap(self.transformer_df, col_name)
        else:
            plot_column = col_name
            plot_df = self.transformer_df
            plot_cmap = self.blue_to_white_to_red
            vmin, vmax = 0, 1

        variable_config = {
            "fixed": ["mlp_expansion_factor", "context_length", "num_dims"],
            "iterate_over": ["checkpoint", "num_tasks"],
            "grid_by": None,
        }
        fixed_values = config["fixed_values"]

        fig, axes = plot_flexible_heatmap(
                df=plot_df,
                variable_config=variable_config,
                fixed_values=fixed_values,
                value_columns=plot_column,
                cmaps=plot_cmap,
                annot=False,
                reverse_x_axis=False,
                reverse_y_axis=True,
                colorbar_location=None,
                figsize=(5, 6),
                vmins=[vmin],
                vmaxs=[vmax],
            )

        fig, axes = plot_main_with_sideplots(
            fig, 
            axes[0][0],
            df=plot_df, 
            variable_config=variable_config, 
            fixed_values=fixed_values, 
            value_columns=[f"distance_from_{algo_name}_{mode}" for algo_name in self.algo_names_dict],
            line_colors=list(self.algo_colors_dict.values()),
            fixed_x_index=config["x_annotated"],
            fixed_y_index=config["y_annotated"],
            top_left_label=self.get_metric_name(distance=True),
            top_right_label="",
            right_top_label="",
            right_bottom_label=self.get_metric_name(distance=True),
            side_plot_label_fontsize=20,
            side_plot_tick_fontsize=12,
        )

        # increase x axis label and tick fontsize
        axes["main"].set_xlabel(self.var_to_title["checkpoint"], fontsize=self.axis_label_fontsize, labelpad=25)  # Increased labelpad
        axes["main"].tick_params(axis='x', labelsize=self.tick_fontsize, pad=8)  # Added pad parameter

        # increase y axis label and tick fontsize
        axes["main"].set_ylabel(self.var_to_title["num_tasks"], fontsize=self.axis_label_fontsize, labelpad=50)  # Increased labelpad
        axes["main"].tick_params(axis='y', labelsize=self.tick_fontsize, pad=2)  # Added pad parameter

        main_ax = axes["main"]

        format_axis_labels_as_powers(main_ax, power_y_offset=0.45)
        format_axis_labels_in_thousands(main_ax, axis="x")

        # Modify x-axis to show only every 3 tick except the last one
        x_ticks = main_ax.get_xticks()
        x_tick_labels = main_ax.get_xticklabels()
        for i, tick in enumerate(x_tick_labels):
            if i % 3 != 0 or i == len(x_tick_labels) - 1:
                tick.set_visible(False)

        top_ax = axes["top"]
        top_ax.set_ylim(top=top_ax.get_ylim()[1]*1.15)

        if save:
            if config.get("custom_name", None) is not None:
                filepath = self.format_plot_name(self.setting, config["fixed_values"], config["custom_name"])
            else:
                filepath = self.format_plot_name(self.setting, config["fixed_values"], f"relative_distance_{mode}")
            self.save_figure(fig, filepath)

        if show:
            plt.show()
            plt.close(fig)

    def generate_prediction_comparison_plot(self, config, show=False, save=True):
        """
        Generate the prediction comparison plot.
        
        Args:
            config (dict): A dictionary of configuration parameters, with the following keys:
                fixed_values: {context_length, mlp_expansion_factor, num_dims}
                fit_results_col [optional]: The column name to plot
                comparison_col [optional]: The column name to compare to
                custom_name: The custom name of the plot (if not provided, the default name will be used)
                prediction_metric [optional]: The metric name to plot
            show (bool): Whether to display the plot
            save (bool): Whether to save the plot
        """
        fixed_values = config["fixed_values"]
        fit_results_col = "bms_results" if config.get("fit_results_col", None) is None else config["fit_results_col"]
        comparison_col = "relative_distance_train" if config.get("comparison_col", None) is None else config["comparison_col"]
        if config.get("prediction_metric", None) is None:
            if self.setting == "linear-regression":
                prediction_metric = "r_squared"
            else:
                prediction_metric = "total_variation_distance"
        else:
            prediction_metric = config["prediction_metric"]

        # Create a custom layout with a full-height heatmap and two stacked smaller heatmaps
        fig = plt.figure(figsize=(6, 6))

        # Create axes with specific positions [left, bottom, width, height]
        # Main plot takes up the left side
        ax1 = fig.add_axes([0.1, 0.1, 0.6, 0.8])

        # Keep the original positions for the right plots
        ax2 = fig.add_axes([0.9, 0.5, 0.4, 0.5333])
        ax3 = fig.add_axes([0.9, -0.15, 0.4, 0.5333])

        # Create a separate axes for the colorbar on the right side
        cbar_ax = fig.add_axes([1.32, -0.15, 0.02, 0.5333])  # [left, bottom, width, height]

        if "train" in comparison_col:
            weights_col = "weights"
        else:
            weights_col = "ood_weights"

        # Handle barycentric weights for the Bayesian Model heatmap
        use_barycentric_bayesian = (len(self.algo_names_dict) > 2)
        if use_barycentric_bayesian:
            # Create a temporary column with the weights from the fit results
            temp_df = self.transformer_df.copy()
            temp_df[f"temp_{weights_col}"] = temp_df[fit_results_col].apply(lambda x: x.get(weights_col))
            
            # Convert barycentric weights to colors
            bayesian_plot_df, bayesian_plot_column, bayesian_plot_cmap, bayesian_vmin, bayesian_vmax = self.convert_barycentric_weights_to_colormap(temp_df, f"temp_{weights_col}")
            bayesian_value_columns = [bayesian_plot_column]
            bayesian_needed_cols = []
        else:
            bayesian_plot_df = self.transformer_df
            bayesian_plot_cmap = self.blue_to_white_to_red
            bayesian_vmin, bayesian_vmax = 0, 1
            bayesian_value_columns = [lambda x: x[fit_results_col].get(weights_col, [float('nan')])[0]]
            bayesian_needed_cols = [fit_results_col]

        # Plot Bayesian Model
        fig1, _ = plot_flexible_heatmap(
            df=bayesian_plot_df,
            variable_config={
                "fixed": ["context_length", "mlp_expansion_factor", "num_dims"],
                "iterate_over": ["checkpoint", "num_tasks"],
                "grid_by": None,
            },
            fixed_values=fixed_values,
            value_columns=bayesian_value_columns,
            needed_cols_for_callable=bayesian_needed_cols,
            ax=ax1,
            annot=False,
            reverse_x_axis=False,
            reverse_y_axis=True,
            cmaps=[bayesian_plot_cmap],
            vmins=[bayesian_vmin],
            vmaxs=[bayesian_vmax],
            titles=["Bayesian Model\n(Eq. 3)"],
            colorbar_labels=[""],
            title_fontsize=self.title_fontsize,
            axis_label_fontsize=self.axis_label_fontsize,
            tick_fontsize=self.tick_fontsize,
            colorbar_location=None,
        )

        # add padding to the plot title
        ax1.set_title("Bayesian Model\n(Eq. 3)", fontsize=self.title_fontsize, pad=30)

        # increase x axis label and tick fontsize
        ax1.set_xlabel(self.var_to_title["checkpoint"], fontsize=self.axis_label_fontsize, labelpad=25)
        ax1.tick_params(axis='x', labelsize=self.tick_fontsize, pad=8)

        # increase y axis label and tick fontsize
        ax1.set_ylabel(self.var_to_title["num_tasks"], fontsize=self.axis_label_fontsize, labelpad=50)
        ax1.tick_params(axis='y', labelsize=self.tick_fontsize, pad=2)

        format_axis_labels_as_powers(ax1)
        format_axis_labels_in_thousands(ax1, axis="x")

        # Modify x-axis to show only every 3rd tick
        x_ticks = ax1.get_xticks()
        x_tick_labels = ax1.get_xticklabels()
        for i, tick in enumerate(x_tick_labels):
            if i % 3 != 0:
                tick.set_visible(False)

        # Get the exact x and y limits from the main plot to ensure identical dimensions
        main_xlim = ax1.get_xlim()
        main_ylim = ax1.get_ylim()

        # Handle barycentric weights for the Transformer heatmap
        mode = "train" if "train" in comparison_col else "eval"
        use_barycentric_transformer = (len(self.algo_names_dict) > 2 and comparison_col == f"barycentric_weights_{mode}")
        
        if use_barycentric_transformer:
            transformer_plot_df, transformer_plot_column, transformer_plot_cmap, transformer_vmin, transformer_vmax = self.convert_barycentric_weights_to_colormap(self.transformer_df, comparison_col)
            transformer_value_columns = [transformer_plot_column]
            transformer_needed_cols = []
            transformer_title = "Transformer\n(Barycentric Weights)"
        else:
            transformer_plot_df = self.transformer_df
            transformer_plot_cmap = self.blue_to_white_to_red
            transformer_vmin, transformer_vmax = 0, 1
            transformer_value_columns = [comparison_col]
            transformer_needed_cols = []
            transformer_title = "Transformer\n(Rel. Distance)"

        # Plot Transformer
        fig2, _ = plot_flexible_heatmap(
            df=transformer_plot_df,
            variable_config={
                "fixed": ["context_length", "mlp_expansion_factor", "num_dims"],
                "iterate_over": ["checkpoint", "num_tasks"],
                "grid_by": None,
            },
            fixed_values=fixed_values,
            value_columns=transformer_value_columns,
            needed_cols_for_callable=transformer_needed_cols,
            ax=ax2,
            annot=False,
            reverse_x_axis=False,
            reverse_y_axis=True,
            cmaps=[transformer_plot_cmap],
            vmins=[transformer_vmin],
            vmaxs=[transformer_vmax],
            titles=[transformer_title],
            colorbar_labels=[""],
            title_fontsize=self.sideplot_title_fontsize,
            axis_label_fontsize=self.sideplot_axis_label_fontsize,
            tick_fontsize=self.sideplot_tick_fontsize,
            x_axis_label="",  # Hide x-label for top right plot
            colorbar_location=None,
        )

        # add padding to the plot title
        ax2.set_title(transformer_title, fontsize=self.sideplot_title_fontsize, pad=20)

        # Force the exact same dimensions as the main plot
        ax2.set_xlim(main_xlim)
        ax2.set_ylim(main_ylim)

        # remove ax2 x and y labels and tick labels
        ax2.set_xlabel("")
        ax2.set_ylabel("")
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])

        greys_cmap = "Greys_r" if self.setting == "linear-regression" else "Greys"

        if "train" in comparison_col:
            distance_col = "id"
        else:
            distance_col = "ood"


        # calculate vmin and vmax based on optimal constant baseline 
        if self.setting == "linear-regression":
            if "prediction_metric" != "mse" :
                if "baseline_results" in self.transformer_df[fit_results_col].iloc[0] and "optimal_constant_baseline" in self.transformer_df[fit_results_col].iloc[0]['baseline_results']:
                    vmin = max(self.transformer_df[fit_results_col].apply(lambda x: x["baseline_results"]["optimal_constant_baseline"][prediction_metric][distance_col]).values.mean(), 0) # if mean gives r^2 < 0, set to 0
                else:
                    vmin = 0
                vmax = 1
            else:
                if "baseline_results" in self.transformer_df[fit_results_col].iloc[0] and "random_weights_baseline" in self.transformer_df[fit_results_col].iloc[0]['baseline_results']:
                    vmax = max(self.transformer_df[fit_results_col].apply(lambda x: x["baseline_results"]["random_weights_baseline"][prediction_metric][distance_col]).values.mean(), 0)
                    print(vmax)
                else: 
                    vmax = 1.5
                vmin=0 
        else:
            vmin = 0
            if "baseline_results" in self.transformer_df[fit_results_col].iloc[0] and "optimal_constant_baseline" in self.transformer_df[fit_results_col].iloc[0]['baseline_results']:
                vmax = self.transformer_df[fit_results_col].apply(lambda x: x["baseline_results"]["optimal_constant_baseline"][prediction_metric][distance_col]).values.mean()
            else:
                vmax = 0.15

        # Plot agreement with next-token prediction - ensure same dimensions as main plot
        fig3, _ = plot_flexible_heatmap(
            df=self.transformer_df,
            variable_config={
                "fixed": ["context_length", "mlp_expansion_factor", "num_dims"],
                "iterate_over": ["checkpoint", "num_tasks"],
                "grid_by": None,
            },
            fixed_values=fixed_values,
            value_columns=[lambda x: x[fit_results_col]['results'][prediction_metric][distance_col]],
            needed_cols_for_callable=[fit_results_col],
            ax=ax3,
            annot=False,
            reverse_x_axis=False,
            reverse_y_axis=True,
            cmaps=[greys_cmap],
            titles=[""],
            colorbar_labels=[""],
            title_fontsize=self.sideplot_title_fontsize,
            axis_label_fontsize=self.sideplot_axis_label_fontsize,
            tick_fontsize=self.sideplot_tick_fontsize,
            colorbar_location=None,  # Don't create a colorbar here
            vmins=[vmin],
            vmaxs=[vmax],
        )


        # Manually create the colorbar for the third plot
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=greys_cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.ax.tick_params(labelsize=self.sideplot_colorbar_fontsize)
        cbar.outline.set_visible(False)

        # add padding to the plot title
        metric = "$R^2$" if self.setting == "linear-regression" else "TV Distance"
        fontsize = self.sideplot_title_fontsize-5 if self.setting == "linear-regression" else self.sideplot_title_fontsize - 7
        title = f"{metric} with Next-Token Predictions"
        ax3.set_title(title, fontsize=fontsize, pad=20) # -5 so it fits

        # Force the exact same dimensions as the main plot
        ax3.set_xlim(main_xlim)
        ax3.set_ylim(main_ylim)

        # remove ax3 x and y labels and tick labels
        ax3.set_xlabel("")
        ax3.set_ylabel("")
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])

        # Add the "Seen" vs "Unseen" boundary line to the R² plot
        # Find boundary between seen and unseen data
        seen_data = self.transformer_df[self.transformer_df[fit_results_col].apply(lambda x: x.get("seen", False) if x is not None else False)]
        unseen_data = self.transformer_df[self.transformer_df[fit_results_col].apply(lambda x: not x.get("seen", False) if x is not None else False)]

        if not seen_data.empty and not unseen_data.empty:
            # Find max checkpoint from seen data and min checkpoint from unseen data
            max_seen_checkpoint = seen_data["checkpoint"].max()
            max_seen_num_tasks = seen_data["num_tasks"].max()
            
            # Get all unique checkpoints and num_tasks for axis positioning
            all_checkpoints = sorted(self.transformer_df["checkpoint"].unique())
            all_num_tasks = sorted(self.transformer_df["num_tasks"].unique())
            
            # Find positions in the plot (index positions)
            max_seen_checkpoint_idx = all_checkpoints.index(max_seen_checkpoint)
            max_seen_num_tasks_idx = all_num_tasks.index(max_seen_num_tasks)
            
            # Draw L-shaped boundary instead of full lines
            # Vertical line from bottom to the intersection point
            ax3.plot([max_seen_checkpoint_idx + 0.5, max_seen_checkpoint_idx + 0.5], 
                    [0, max_seen_num_tasks_idx + 0.5], 
                    color='black', linestyle='--', linewidth=2)
            
            # Horizontal line from left to the intersection point
            ax3.plot([0, max_seen_checkpoint_idx + 0.5], 
                    [max_seen_num_tasks_idx + 0.5, max_seen_num_tasks_idx + 0.5], 
                    color='black', linestyle='--', linewidth=2)
            
            # Add "Seen" and "Unseen" labels
            ax3.text(max_seen_checkpoint_idx/2, max_seen_num_tasks_idx*1/6, 
                    "Used for Fitting", ha='center', va='center', fontsize=self.tick_fontsize-3)
            ax3.text(max_seen_checkpoint_idx/3.7, 
                    (len(all_num_tasks) + max_seen_num_tasks_idx)/2, 
                    "Unseen", ha='center', va='center', fontsize=self.tick_fontsize-3)
       
        if save:
            if config.get("custom_name", None) is not None:
                filepath = self.format_plot_name(self.setting, config["fixed_values"], config["custom_name"])
            else:
                filepath = self.format_plot_name(self.setting, config["fixed_values"], f"prediction_comparison")
            self.save_figure(fig, filepath)

        if show:
            plt.show()
            plt.close(fig)

    def generate_posterior_odds_plot(self, config, show=False, save=True):
        """
        Generate the posterior odds plot.
        
        Args:
            config (dict): A dictionary of configuration parameters, with the following keys:
                fixed_values: {context_length, mlp_expansion_factor, num_dims}
                custom_name: The custom name of the plot (if not provided, the default name will be used)
            show (bool): Whether to display the plot
            save (bool): Whether to save the plot
        """
        fixed_values = config["fixed_values"]
        tick_fontsize = 17

        # find the max and min values of the log prior odds, log bayes factor, and log posterior odds
        min_log_prior_odds = self.transformer_df["bms_results"].apply(lambda x: x.get("log_prior_odds_id", float('nan'))).min()
        max_log_bayes_factor = self.transformer_df["bms_results"].apply(lambda x: x.get("log_bayes_factor_id", float('nan'))).max()
        abs_distance_max = max(abs(min_log_prior_odds), abs(max_log_bayes_factor))

        # Plot posterior odds
        fig, ax = plot_flexible_heatmap(
            df=self.transformer_df,
            variable_config={
                "fixed": ["context_length", "mlp_expansion_factor", "num_dims"],
                "iterate_over": ["checkpoint", "num_tasks"],
                "grid_by": None,
            },
            fixed_values=fixed_values,
            value_columns=[lambda x: x["bms_results"].get("log_prior_odds_id", float('nan')),
                        lambda x: x["bms_results"].get("log_bayes_factor_id", float('nan')),
                        lambda x: x["bms_results"].get("log_posterior_odds_id", float('nan')),
                        ],
            needed_cols_for_callable=["bms_results"],
            annot=False,
            reverse_x_axis=False,
            reverse_y_axis=True,
            cmaps=[self.blue_to_white, self.white_to_red, self.blue_to_white_to_red],
            titles=["Log Prior Odds", "Log Bayes Factor", "Log Posterior Odds"],
            colorbar_labels=[""],
            title_fontsize=self.title_fontsize,
            axis_label_fontsize=self.axis_label_fontsize,
            tick_fontsize=tick_fontsize,
            vmins=[-abs_distance_max, 0, -abs_distance_max],
            vmaxs=[0, abs_distance_max, abs_distance_max],
            figsize=(17, 5)
        )

        # convert labels on leftmost plot to powers of 2
        format_axis_labels_as_powers(ax[0][0], base_x_pos=-0.12, power_y_offset=0.7)

        # remove y axis labels and ticks on center and leftmost plots
        ax[0][1].set_ylabel("")
        ax[0][1].set_yticklabels([])
        ax[0][2].set_ylabel("")
        ax[0][2].set_yticklabels([])

        # increase y axis label and tick fontsize
        ax[0][0].set_ylabel(self.var_to_title["num_tasks"], fontsize=self.axis_label_fontsize, labelpad=35)
        ax[0][0].tick_params(axis='y', labelsize=tick_fontsize, pad=2)

        # set x axis label on all plots
        for i in range(3):
            ax[0][i].set_xlabel(self.var_to_title["checkpoint"], fontsize=self.axis_label_fontsize, labelpad=25)
            ax[0][i].tick_params(axis='x', labelsize=tick_fontsize, pad=8)

            # format x axis labels in thousands
            format_axis_labels_in_thousands(ax[0][i], axis="x")

            # only keep every 3rd tick label
            x_tick_labels = ax[0][i].get_xticklabels()
            for i, tick in enumerate(x_tick_labels):
                if i % 3 != 0:
                    tick.set_visible(False)

        # set padding for the titles of all plots
        ax[0][0].set_title("Log Prior Odds", fontsize=self.title_fontsize, pad=40)
        ax[0][1].set_title("Log Bayes Factor", fontsize=self.title_fontsize, pad=40)
        ax[0][2].set_title("Log Posterior Odds", fontsize=self.title_fontsize, pad=40)

        if save:
            if config.get("custom_name", None) is not None:
                filepath = self.format_plot_name(self.setting, config["fixed_values"], config["custom_name"])
            else:
                filepath = self.format_plot_name(self.setting, config["fixed_values"], f"posterior_odds")
            self.save_figure(fig, filepath)

        if show:
            plt.show()
            plt.close(fig)

    def generate_algorithm_probabilities_plot(self, config, show=False, save=True):
        """
        Generate the algorithm probabilities plot.
        
        Args:
            config (dict): A dictionary of configuration parameters, with the following keys:
                fixed_values: {context_length, mlp_expansion_factor, num_dims}
                task_diversity_values: The task diversity values to plot
                fit_results_col: The column name of the fit results to plot
                custom_name: The custom name of the plot (if not provided, the default name will be used)
            show (bool): Whether to display the plot
            save (bool): Whether to save the plot
        """
        # List of task diversity values to plot
        task_diversity_values = config["task_diversity_values"]
        fixed_values = config["fixed_values"]
        current_fixed_values = fixed_values.copy()
        fit_results_col = config.get("fit_results_col", "bms_results")

        fig, axes = plt.subplots(1, len(task_diversity_values), figsize=(18, 3.5), sharey=True)

        if len(self.algo_names_dict) < 3:
            legend_loc = "center right"
        else:
            legend_loc = "center left"


        # Create plots for each task diversity value
        for i, num_tasks in enumerate(task_diversity_values):
            # Update fixed values for this subplot
            current_fixed_values["num_tasks"] = num_tasks
            
            # Call plot_flexible_scatterplot for this subplot
            _, ax = plot_flexible_scatterplot(
                df=self.transformer_df,
                variable_config={
                    "fixed": ["context_length", "mlp_expansion_factor", "num_dims", "num_tasks"],
                    "iterate_over": "checkpoint",
                    "grid_by": None,
                },
                fixed_values=current_fixed_values,
                value_column=[lambda x, idx=algo_idx: x[fit_results_col].get(f"weights", float('nan'))[idx] for algo_idx in range(len(self.algo_names_dict))],
                needed_cols_for_callable=[fit_results_col],
                x_scale="log",
                x_scale_base=10,
                y_label="Posterior Probability" if i == 0 else "", 
                color=[self.algo_colors_dict[algo_name] for algo_name in self.algo_names_dict],
                linestyle=["-", "-"],
                marker=["", ""],
                legend_labels=[self.algo_names_dict[algo_name] for algo_name in self.algo_names_dict], 
                legend_loc=legend_loc,
                show_title=True,
                frameon=False,
                axis_label_fontsize=self.tick_fontsize,
                tick_fontsize=self.tick_fontsize,
                legend_fontsize=self.tick_fontsize,
                title_fontsize=self.tick_fontsize,
                ax=axes[i]  # Use the corresponding subplot
            )

            if i != 0:
                axes[i].legend_.remove()
            
            # Add task diversity as title to each subplot
            axes[i].set_title(f"{self.var_to_title['num_tasks']} = {num_tasks}", fontsize=self.tick_fontsize)
            
            # Set x-axis label
            axes[i].set_xlabel(self.var_to_title["checkpoint"], fontsize=self.tick_fontsize)

            # remove minor ticks from x axis
            axes[i].xaxis.set_minor_locator(plt.NullLocator())

        if save:
            if config.get("custom_name", None) is not None:
                filepath = self.format_plot_name(self.setting, config["fixed_values"], config["custom_name"])
            else:
                filepath = self.format_plot_name(self.setting, config["fixed_values"], f"algo_probabilities")
            self.save_figure(fig, filepath)

        if show:
            plt.show()
            plt.close(fig)

    def generate_sublinear_evidence_accumulation_plot(self, config, show=False, save=True):
        """
        Generate the sublinear evidence accumulation plot.
        
        Args:
            config (dict): A dictionary of configuration parameters, with the following keys:
                fixed_values: {context_length, mlp_expansion_factor, num_dims}
                compare_with_predicted: Whether to compare with predicted values
                tasks_to_label: The task diversity values to label
                custom_name: The custom name of the plot (if not provided, the default name will be used)
            show (bool): Whether to display the plot
            save (bool): Whether to save the plot
        """
        compare_with_predicted = config["compare_with_predicted"]

        # Create appropriate figure size and grid based on comparison mode
        if compare_with_predicted:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12), gridspec_kw={'width_ratios': [1, 1]})
        else:
            fig, axes = plt.subplots(2, 1, figsize=(7, 12))
            # Convert to 2D array format for consistent indexing later
            axes = np.array(axes).reshape(-1, 1)
 
        cmap = self.blue_to_white_to_red  
        # Define color normalization range
        norm = plt.Normalize(vmin=0, vmax=1)

        # Fixed values for both plots
        fixed_values = config["fixed_values"]

        # Filter data based on fixed values
        filtered_df = self.transformer_df.query(
            "context_length == @fixed_values['context_length'] & "
            "mlp_expansion_factor == @fixed_values['mlp_expansion_factor'] & "
            "num_dims == @fixed_values['num_dims']", engine="python"
        )

        # Extract alpha from the first row that has bms_results
        alpha = None
        for _, row in filtered_df.iterrows():
            if row.get('bms_results') and 'params' in row['bms_results'] and 'log_alpha' in row['bms_results']['params']:
                alpha = np.exp(row['bms_results']['params']['log_alpha'])
                break

        if alpha is None:
            alpha = 0

        print(f"Using alpha = {alpha}, plotting against N^(1-{alpha}) = N^{1-alpha}")

        # Get unique num_tasks values
        num_tasks_values = sorted(filtered_df['num_tasks'].unique())

        # Choose which tasks to label - exclude D=16 since it's too close to D=4
        tasks_to_label = config["tasks_to_label"] if config["tasks_to_label"] is not None else [4, 64, 256, 1024]

        # define logistic function to be fit
        logistic = lambda x, x0, k, l: l / (1 + np.exp(-k * (x - x0)))

        # Create these plots first to determine y-axis limits

        # BOTTOM LEFT PLOT: Relative distance vs N^(1-alpha)
        logistic_fit_params = {}
        for num_tasks in tqdm_func()(num_tasks_values):
            task_data = filtered_df[filtered_df['num_tasks'] == num_tasks].sort_values('checkpoint')
            
            if len(task_data) < 3:  # Skip if not enough data points
                continue
            
            x_data = task_data['checkpoint'].values
            x_data_transformed = x_data ** (1 - alpha)  # Transform to N^(1-alpha)
            y_data = task_data['relative_distance_train'].values            
                        
            # Plot the transformed data
            axes[1, 0].plot(x_data_transformed, y_data, color='black', linestyle='', linewidth=1, alpha=0.3)
            axes[1, 0].scatter(
                x_data_transformed, y_data, 
                c=y_data,  
                cmap=cmap,
                norm=norm,
                s=50, 
                alpha=0.9, 
                edgecolor='black', 
                linewidth=0.5
            )

            if alpha != 0:
                # fit logistic function
                p0 = [np.median(x_data_transformed), 1.0, np.median(y_data)]
                popt, _ = curve_fit(logistic, x_data_transformed, y_data, p0=p0, maxfev=100000)
                x0, k, l = popt
                logistic_predictions = logistic(x_data_transformed, x0, k, l)
                logistic_fit_params[num_tasks] = (x0, k, l)

                # plot logistic predictions
                axes[1, 0].plot(x_data_transformed, logistic_predictions, color='black', linestyle='-', linewidth=2, alpha=0.4)

            
            # Add task diversity label at the end of the curve
            if num_tasks in tasks_to_label and len(x_data) > 0:
                axes[1, 0].text(
                    x_data_transformed[-1] * 1.02, 
                    y_data[-1], 
                    f"$D$={num_tasks}", 
                    fontsize=self.sideplot_legend_fontsize, ha='left', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                )

        # Configure bottom left plot
        axes[1, 0].set_xlabel(f'$N^{{1-\\alpha}}$', fontsize=self.axis_label_fontsize)
        axes[1, 0].set_ylabel('Relative Distance', fontsize=self.axis_label_fontsize)
        axes[1, 0].tick_params(axis='both', labelsize=self.tick_fontsize)

        # TOP LEFT PLOT: Relative distance vs Training Steps (N)
        for num_tasks in tqdm_func()(num_tasks_values):
            task_data = filtered_df[filtered_df['num_tasks'] == num_tasks].sort_values('checkpoint')
            
            if len(task_data) < 3:  # Skip if not enough data points
                continue
            
            x_data = task_data['checkpoint'].values
            y_data = task_data['relative_distance_train'].values
            
            # Plot the original data
            axes[0, 0].plot(x_data, y_data, color='black', linestyle='', linewidth=1, alpha=0.3)
            axes[0, 0].scatter(
                x_data, y_data, 
                c=y_data,  
                cmap=cmap,
                norm=norm,
                s=50, 
                alpha=0.9, 
                edgecolor='black', 
                linewidth=0.5
            )

            if alpha != 0:
                # get logistic predictions
                x0, k, l = logistic_fit_params[num_tasks]
                transformed_x_data = x_data ** (1 - alpha)
                logistic_predictions = logistic(transformed_x_data, x0, k, l)
                axes[0, 0].plot(x_data, logistic_predictions, color='black', linestyle='-', linewidth=2, alpha=0.4)
                        
            # Add task diversity label at the end of the curve
            if num_tasks in tasks_to_label and len(x_data) > 0:
                axes[0, 0].text(
                    x_data[-1] * 1.02, 
                    y_data[-1], 
                    f"$D$={num_tasks}", 
                    fontsize=self.sideplot_legend_fontsize, ha='left', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                )

        # Configure top left plot
        axes[0, 0].set_xlabel(self.var_to_title["checkpoint"], fontsize=self.axis_label_fontsize)
        axes[0, 0].set_ylabel('Relative Distance', fontsize=self.axis_label_fontsize)
        axes[0, 0].tick_params(axis='both', labelsize=self.tick_fontsize)
        # Only add "Measured" title when comparing
        if compare_with_predicted:
            axes[0, 0].set_title('Measured', fontsize=self.axis_label_fontsize, pad=20)
        format_axis_labels_in_thousands(axes[0, 0], axis='x')

        # Add horizontal line at y=0.5 for both left plots
        axes[0, 0].axhline(y=0.5, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
        axes[1, 0].axhline(y=0.5, color='black', linestyle='--', alpha=0.7, linewidth=1.5)

        # Store the y-axis limits from the left plots
        top_left_ylim = axes[0, 0].get_ylim()
        bottom_left_ylim = axes[1, 0].get_ylim()

        # Only create the predicted plots if compare_with_predicted is True
        if compare_with_predicted:
            # TOP RIGHT PLOT: Predicted values vs Training Steps (N)
            for num_tasks in num_tasks_values:
                task_data = filtered_df[filtered_df['num_tasks'] == num_tasks].sort_values('checkpoint')
                
                if len(task_data) < 3:  # Skip if not enough data points
                    continue
                
                x_data = task_data['checkpoint'].values
                
                # Get predicted values from bms_results
                y_data = task_data['bms_results'].apply(lambda x: x.get('weights', [0])[0] if x is not None else 0).values
                
                # Plot the predicted data
                axes[0, 1].plot(x_data, y_data, color='black', linestyle='-', linewidth=1, alpha=0.3)
                axes[0, 1].scatter(
                    x_data, y_data, 
                    c=y_data,  
                    cmap=cmap,
                    norm=norm,
                    s=50, 
                    alpha=0.9, 
                    edgecolor='black', 
                    linewidth=0.5
                )
                
                # Add task diversity label at the end of the curve for selected task values
                if num_tasks in tasks_to_label and len(x_data) > 0:
                    axes[0, 1].text(
                        x_data[-1] * 1.01, 
                        y_data[-1], 
                        f"$D$={num_tasks}", 
                        fontsize=self.sideplot_legend_fontsize, ha='left', va='center',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                    )

            # Configure top right plot
            axes[0, 1].set_xlabel(self.var_to_title["checkpoint"], fontsize=self.axis_label_fontsize)
            axes[0, 1].tick_params(axis='both', labelsize=self.tick_fontsize)
            axes[0, 1].set_title('Predicted', fontsize=self.axis_label_fontsize, pad=20)
            # Set identical y-axis limits as the top left plot
            axes[0, 1].set_ylim(top_left_ylim)
            format_axis_labels_in_thousands(axes[0, 1], axis='x')

            # BOTTOM RIGHT PLOT: Predicted values vs N^(1-alpha)
            for num_tasks in num_tasks_values:
                task_data = filtered_df[filtered_df['num_tasks'] == num_tasks].sort_values('checkpoint')
                
                if len(task_data) < 3:  # Skip if not enough data points
                    continue
                
                x_data = task_data['checkpoint'].values
                x_data_transformed = x_data ** (1 - alpha)  # Transform to N^(1-alpha)
                
                # Get predicted values from bms_results
                y_data = task_data['bms_results'].apply(lambda x: x.get('weights', [0])[0] if x is not None else 0).values
                
                # Plot the transformed predicted data
                axes[1, 1].plot(x_data_transformed, y_data, color='black', linestyle='-', linewidth=1, alpha=0.3)
                axes[1, 1].scatter(
                    x_data_transformed, y_data, 
                    c=y_data,  
                    cmap=cmap,
                    norm=norm,
                    s=50, 
                    alpha=0.9, 
                    edgecolor='black', 
                    linewidth=0.5
                )
                
                # Add task diversity label at the end of the curve for selected task values
                if num_tasks in tasks_to_label and len(x_data) > 0:
                    axes[1, 1].text(
                        x_data_transformed[-1] * 1.01, 
                        y_data[-1], 
                        f"$D$={num_tasks}", 
                        fontsize=self.sideplot_legend_fontsize, ha='left', va='center',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                    )

            # Configure bottom right plot
            axes[1, 1].set_xlabel(f'$N^{{1-\\alpha}}$', fontsize=self.axis_label_fontsize)
            axes[1, 1].tick_params(axis='both', labelsize=self.tick_fontsize)
            # Set identical y-axis limits as the bottom left plot
            axes[1, 1].set_ylim(bottom_left_ylim)

            # Add horizontal line at y=0.5 for both right plots
            axes[0, 1].axhline(y=0.5, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
            axes[1, 1].axhline(y=0.5, color='black', linestyle='--', alpha=0.7, linewidth=1.5)

            # Ensure x-axis limits match between measured and predicted plots
            axes[0, 1].set_xlim(axes[0, 0].get_xlim())
            axes[1, 1].set_xlim(axes[1, 0].get_xlim())
            
        # Add space between columns for comparison layout
        plt.subplots_adjust(wspace=0.5)
        plt.subplots_adjust(hspace=0.5)
        plt.tight_layout()

        if save:
            if config.get("custom_name", None) is not None:
                filepath = self.format_plot_name(self.setting, config["fixed_values"], config["custom_name"])
            else:
                filepath = self.format_plot_name(self.setting, config["fixed_values"], f"sublinear_evidence_accumulation")
            self.save_figure(fig, filepath)
        if show:
            plt.show()
            plt.close(fig)


    def generate_transience_predictions_plot(self, config, show=False, save=True):
        """
        Generate the transience predictions plot.

        Args:
            config (dict): A dictionary of configuration parameters, with the following keys:
                fixed_values: {context_length, mlp_expansion_factor, num_dims}
                two_hypotheses_cutoff: The cutoff for modeling dynamics with two hypotheses
                close_to_half_threshold: The threshold for considering a relative distance to be close to 0.5
                custom_name: The custom name of the plot (if not provided, the default name will be used)
            show (bool): Whether to display the plot
            save (bool): Whether to save the plot
        """
        setting = config["fixed_values"]
        # Filter the dataframe first and keep all columns
        filtered_df = self.transformer_df.query("context_length == @setting['context_length'] & mlp_expansion_factor == @setting['mlp_expansion_factor'] & num_dims == @setting['num_dims']", engine="python").copy()
        close_to_half_threshold = config.get("close_to_half_threshold", 0.05)

        # get fitted params
        fitted_params = filtered_df["bms_results"].iloc[0]["params"]
        beta = np.exp(fitted_params["log_beta"])
        alpha = np.exp(fitted_params["log_alpha"])
        gamma = np.exp(fitted_params["log_gamma"])

        fitted_algo_df = self.algo_df.query("num_dims == @setting['num_dims'] & context_length == @setting['context_length']", engine="python")

        # Calculate N* for each num_tasks value
        n_star_values = []
        num_tasks_values = []
        numerator_values = []
        denominator_values = []

        for _, row in fitted_algo_df.iterrows():
            # Extract values
            k_mem = row["memorized_complexity"]
            k_gen = row["generalized_complexity"]
            mem_nll = row["memorized_id_nll"]
            gen_nll = row["generalized_id_nll"]
            
            # Calculate N* using the formula
            numerator = (k_gen**beta - k_mem**beta)*np.log(2)
            denominator = gamma * (mem_nll - gen_nll)
            n_star = (numerator / denominator)**(1/(1-alpha))

            numerator_values.append(numerator)
            denominator_values.append(mem_nll - gen_nll)
            n_star_values.append(n_star)
            num_tasks_values.append(row["num_tasks"])

        # Convert to numpy arrays
        n_star_values = np.array(n_star_values)
        num_tasks_values = np.array(num_tasks_values)
        numerator_values = np.array(numerator_values)
        denominator_values = np.array(denominator_values)

        # Find actual transition points (where relative distance ≈ 0.5)
        real_transition_checkpoints = []
        real_transition_num_tasks = []
        transition_ranges = []  # Store the range of checkpoints for each transition

        # Filter the dataframe to get only the rows with the same dims
        filtered_transformer_df = self.transformer_df.query(f"num_dims == {setting['num_dims']} & context_length == {setting['context_length']} & mlp_expansion_factor == {setting['mlp_expansion_factor']}", engine="python")

        # Group by num_tasks to find transition checkpoint for each task diversity value
        for num_tasks, group in filtered_transformer_df.groupby('num_tasks'):
            # Sort by checkpoint
            group = group.sort_values('checkpoint')
            
            # Only process if relative_distance_train exists and has values close to 0.5
            if 'relative_distance_train' in group.columns:
                # Find values between 0.4 and 0.6
                in_range_mask = (group['relative_distance_train'] >= 0.4) & (group['relative_distance_train'] <= 0.6)
                if in_range_mask.any():
                    # Get the checkpoint closest to 0.5
                    closest_to_half_idx = np.abs(group['relative_distance_train'] - 0.5).idxmin()
                    closest_checkpoint = group.loc[closest_to_half_idx, 'checkpoint']
                    
                    # Get the min and max checkpoints for values in range
                    min_checkpoint = group.loc[in_range_mask, 'checkpoint'].min()
                    max_checkpoint = group.loc[in_range_mask, 'checkpoint'].max()
                    
                    real_transition_checkpoints.append(closest_checkpoint)
                    real_transition_num_tasks.append(num_tasks)
                    transition_ranges.append((min_checkpoint, max_checkpoint))

        # Create a mask to filter predictions:
        # Keep only points that are either above threshold OR have a corresponding actual value
        keep_mask = np.zeros_like(n_star_values, dtype=bool)
        for i, (n_star, num_task) in enumerate(zip(n_star_values, num_tasks_values)):
            # Keep if above threshold
            if n_star >= config["two_hypotheses_cutoff"]:
                keep_mask[i] = True
            # Or keep if we have an actual transition point for this task value
            elif num_task in real_transition_num_tasks:
                keep_mask[i] = True

        # Apply the mask to filter the data
        filtered_n_star = n_star_values[keep_mask]
        filtered_num_tasks = num_tasks_values[keep_mask]
        filtered_numerator = numerator_values[keep_mask]
        filtered_denominator = denominator_values[keep_mask]

        # Filter out any invalid values for the side plots
        valid_mask = (filtered_numerator != 0) & (filtered_denominator != 0)
        valid_numerator = filtered_numerator[valid_mask]
        valid_denominator = filtered_denominator[valid_mask] 
        valid_num_tasks = filtered_num_tasks[valid_mask]

        # Create a figure with a custom layout
        fig = plt.figure(figsize=(10, 10))

        # Main plot takes most of the space
        main_ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])

        # Add two small plots on right with increased spacing
        delta_k_ax = fig.add_axes([0.84, 0.55, 0.2, 0.35])  # Top plot
        delta_l_ax = fig.add_axes([0.84, 0.1, 0.2, 0.35])   # Bottom plot with more vertical space from top plot

        # Plot main scatter plot with filtered data
        main_ax.scatter(filtered_num_tasks, filtered_n_star, color='black', alpha=0.7, s=50, label='Predicted $N^{*}$')

        if real_transition_checkpoints:
            # Plot the mean points
            main_ax.scatter(real_transition_num_tasks, real_transition_checkpoints, 
                            marker='X', s=150, 
                            facecolor='white', edgecolor='black', linewidth=1.5, alpha=0.5,
                            label='Actual (Rel. dist. closest to 0.5, Range 0.4-0.6)' )

            # Plot the range lines
            for (num_tasks, checkpoint, (min_checkpoint, max_checkpoint)) in zip(real_transition_num_tasks, real_transition_checkpoints, transition_ranges):
                main_ax.plot([num_tasks, num_tasks], [min_checkpoint, max_checkpoint], 
                            color='grey', alpha=0.6, linewidth=2, linestyle='-')

        # Add equation annotation to the main plot
        main_ax.annotate('$N^{*}(D) = \\left[ \\frac{\\Delta K(D)^{\\beta}}{\\gamma \\Delta L(D)} \\right]^{\\frac{1}{1-\\alpha}}$',
                        xy=(0.34, 0.83), xycoords='axes fraction',
                        fontsize=self.sideplot_title_fontsize+3, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7, edgecolor='none'))

        # Add horizontal line for two hypotheses threshold
        threshold = filtered_df["approximate_interpolation_threshold"].iloc[0]
        main_ax.axhline(y=threshold, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Add annotation for the threshold
        middle_task = filtered_num_tasks[len(filtered_num_tasks) // 2]
        main_ax.annotate('Two hypotheses threshold', 
                        xy=(2**6, threshold * 1.1),
                        fontsize=self.tick_fontsize, ha='left', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

        # Increase font sizes and padding
        main_ax.set_xlabel(self.var_to_title["num_tasks"], fontsize=self.axis_label_fontsize, labelpad=30)
        main_ax.set_ylabel('Transience Training Step $N^{*}$', fontsize=self.axis_label_fontsize, labelpad=30)
        main_ax.legend(fontsize=self.legend_fontsize-5, loc='upper left', frameon=False, bbox_to_anchor=(-0.03, 1.0))
        main_ax.set_yscale('log')
        main_ax.set_xscale('log', base=2)
        main_ax.grid(False)
        main_ax.tick_params(labelsize=self.tick_fontsize)
        main_ax.minorticks_off()

        # Plot delta K (numerator) at bottom right
        delta_k_ax.scatter(valid_num_tasks, np.abs(valid_numerator), color=self.blue, alpha=0.7, s=40)
        delta_k_ax.set_ylabel('$\\Delta K(D)^{\\beta}$', fontsize=self.sideplot_title_fontsize, labelpad=1)
        delta_k_ax.set_xscale('log', base=2)
        delta_k_ax.set_yscale('log', base=2)
        delta_k_ax.grid(False)
        delta_k_ax.tick_params(labelsize=self.sideplot_tick_fontsize-2)

        # Plot delta L (denominator) at top right
        delta_l_ax.scatter(valid_num_tasks, np.abs(valid_denominator), color=self.red, alpha=0.7, s=40)
        delta_l_ax.set_xlabel(self.var_to_title["num_tasks"], fontsize=20, labelpad=8)
        delta_l_ax.set_ylabel('$\\gamma \\Delta L(D)$', fontsize=self.sideplot_title_fontsize, labelpad=2)
        delta_l_ax.set_xscale('log', base=2)
        delta_l_ax.grid(False)
        delta_l_ax.tick_params(labelsize=self.sideplot_tick_fontsize-2)
        
        # show every second tick label on the y axis
        for i, tick in enumerate(delta_l_ax.yaxis.get_major_ticks()):
            if i % 2 != 0:
                tick.tick1line.set_visible(False)  # left tick line
                tick.label1.set_visible(False)     # label on left

        if save:
            if config.get("custom_name", None) is not None:
                filepath = self.format_plot_name(self.setting, config["fixed_values"], config["custom_name"])
            else:
                filepath = self.format_plot_name(self.setting, config["fixed_values"], f"transience_predictions")
            self.save_figure(fig, filepath)
        if show:
            plt.show()
            plt.close(fig)


    def generate_vector_field_plot(self, config, show=False, save=True):
        """
        Generate a vector field plot.

        Args:
            config (dict): A dictionary of configuration parameters, with the following keys:
                fixed_values: {context_length, mlp_expansion_factor, num_dims}
                fixed_x_index: index of the x axis to fix
                fixed_y_index: index of the y axis to fix
                custom_name: The custom name of the plot (if not provided, the default name will be used)
            show (bool): Whether to display the plot
            save (bool): Whether to save the plot
        """
        # Setup plot configuration
        variable_config={
            "fixed": ["context_length", "mlp_expansion_factor", "num_dims"],
            "iterate_over": ["checkpoint", "num_tasks"],
            "grid_by": "",
        }
        fixed_values = config["fixed_values"]

        # compute derivatives
        # derivative over D 
        self.transformer_df = compute_derivatives(df=self.transformer_df, 
                                            column_name="relative_distance_train", 
                                            group_by_vars=["context_length", "mlp_expansion_factor", "num_dims", "checkpoint"], 
                                            vary_by_var="num_tasks",use_log_scale=True, result_suffix="D")

        # derivative over N
        self.transformer_df = compute_derivatives(df=self.transformer_df, 
                                            column_name="relative_distance_train", 
                                            group_by_vars=["context_length", "mlp_expansion_factor", "num_dims", "num_tasks"], 
                                            vary_by_var="checkpoint",use_log_scale=True, result_suffix="N")

        # Call the function with adjusted parameters to handle the large horizontal vector
        fig, axes = plot_vector_field(
            self.transformer_df, 
            fixed_values=fixed_values,
            figsize=(5, 6),
            sqrt_scale_x_axis=True,
            x_min=(np.sqrt(self.transformer_df.query("num_dims == @fixed_values['num_dims'] and context_length == @fixed_values['context_length'] and mlp_expansion_factor == @fixed_values['mlp_expansion_factor']", engine="python")["checkpoint"].min())-7)**2,
            x_max=(np.sqrt(self.transformer_df.query("num_dims == @fixed_values['num_dims'] and context_length == @fixed_values['context_length'] and mlp_expansion_factor == @fixed_values['mlp_expansion_factor']", engine="python")["checkpoint"].max())+5)**2,
            y_min=1.7,
            y_max=12.3,
            background_cmap=self.blue_to_white_to_red,
            blue_color=self.blue,
            red_color=self.red,
            scale=30,
        )

        fig, axes = plot_main_with_sideplots(
            fig, 
            axes,
            df=self.transformer_df, 
            variable_config=variable_config, 
            fixed_values=fixed_values, 
            value_columns=[["relative_distance_train_second_derivative_over_N"], ["relative_distance_train_second_derivative_over_D"]],
            line_colors=["black"],
            fixed_x_index=config["fixed_x_index"],
            fixed_y_index=config["fixed_y_index"],
            top_left_label=f"$\\partial^2 / \\partial N^2$",
            top_right_label="",
            right_top_label="",
            right_bottom_label="$\\partial^2 / \\partial D^2$",
            side_plot_label_fontsize=20,
            side_plot_tick_fontsize=10,
            is_heatmap=False,
            right_plot_ax_lim_scaling=config.get("right_plot_ax_lim_scaling", 1.35),
            top_plot_ax_lim_scaling=config.get("top_plot_ax_lim_scaling", 1.25)
        )        


        tick_fontsize = self.tick_fontsize
        label_fontsize = self.axis_label_fontsize

        # increase x axis label and tick fontsize
        axes["main"].set_xlabel(self.var_to_title["checkpoint"], fontsize=label_fontsize, labelpad=25)  # Increased labelpad
        axes["main"].tick_params(axis='x', labelsize=tick_fontsize, pad=8)  # Added pad parameter

        # increase y axis label and tick fontsize
        axes["main"].set_ylabel(self.var_to_title["num_tasks"], fontsize=label_fontsize, labelpad=50)  # Increased labelpad
        axes["main"].tick_params(axis='y', labelsize=tick_fontsize, pad=2)  # Added pad parameter

        # add reference line at 0 in each side plot
        axes["top"].axhline(y=0, color='black', linestyle='--', alpha=1, linewidth=1)
        axes["right"].axvline(x=0, color='black', linestyle='--', alpha=1, linewidth=1)


        # Modify x-axis to show only every 4th tick
        main_ax = axes["main"]
        format_axis_labels_in_thousands(main_ax, axis="x")
        x_tick_labels = main_ax.get_xticklabels()
        for i, tick in enumerate(x_tick_labels):
            if i % 4 != 0:
                tick.set_visible(False)

        if save:
            if config.get("custom_name", None) is not None:
                filepath = self.format_plot_name(self.setting, config["fixed_values"], config["custom_name"])
            else:
                filepath = self.format_plot_name(self.setting, config["fixed_values"], f"vector_field")
            self.save_figure(fig, filepath)
        if show:
            plt.show()
            plt.close(fig)

    def generate_titration_plot(self, config, show=False, save=True):
        """
        Generate a titration plot with multiple rows, each corresponding to a different metric.
        
        Args:
            config (dict): Configuration dictionary with the following keys:
                col_names (list): List of column names to plot, each creating a new row
                row_titles (list, optional): Titles for each row, displayed on leftmost plot
                mode (str, optional): 'train' or 'eval' mode
                variable_config (dict): Configuration for plot variables
                fixed_values (dict): Fixed parameter values
                task_diversity_values (list, optional): List of task diversity values to include
                hidden_size (int, optional): Hidden size, if mlp_expansion_factor is being titrated
                custom_name: The custom name of the plot (if not provided, the default name will be used)
            show (bool): Whether to display the plot
            save (bool): Whether to save the plot
        """
        mode = "train" if config.get("mode", None) is None else config["mode"]
        
        # Get column names from config, default to a single relative_distance column if not provided
        if "col_names" in config:
            col_names = config["col_names"]
        else:
            col_names = [f"relative_distance_{mode}"]
        
        # Get row titles if provided
        row_titles = config.get("row_titles", [None] * len(col_names))
        if len(row_titles) < len(col_names):
            row_titles.extend([None] * (len(col_names) - len(row_titles)))
        
        variable_config = config["variable_config"].copy()
        # Extract grid_by variable before removing from variable_config
        grid_by_var = variable_config.pop("grid_by", None)
        fixed_values = config["fixed_values"]

        # Filter transformer df to only include rows where num_tasks meets criteria
        task_diversity_values = [32, 64, 128, 256] if config.get("task_diversity_values", None) is None else config["task_diversity_values"]
        filtered_df = self.transformer_df.query("num_tasks in @task_diversity_values", engine="python")
        
        # If we have a grid_by variable, get its unique values
        if grid_by_var:
            grid_values = sorted(filtered_df[grid_by_var].unique())
            num_cols = len(grid_values)
        else:
            grid_values = [None]
            num_cols = 1
        
        # Create figure with appropriate number of rows and columns
        num_rows = len(col_names)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 5 * num_rows), sharey=True)
        
        # Ensure axes is always a 2D array
        if num_rows == 1 and num_cols == 1:
            axes = np.array([[axes]])
        elif num_rows == 1:
            axes = np.array([axes]).reshape(1, -1)
        elif num_cols == 1:
            axes = np.array(axes).reshape(-1, 1)
        else:
            axes = np.array(axes)
        
        # Create separate heatmaps for each grid value (column)
        for col_idx, grid_value in enumerate(grid_values):
            # Filter data for this grid value if applicable
            if grid_value is not None:
                grid_df = filtered_df[filtered_df[grid_by_var] == grid_value].copy()
                
                # Add grid_by_var to fixed_values temporarily
                temp_fixed_values = fixed_values.copy()
                temp_fixed_values[grid_by_var] = grid_value
            else:
                grid_df = filtered_df.copy()
                temp_fixed_values = fixed_values.copy()
            
            # Get the axes for this column
            if num_cols == 1:
                column_axes = axes[:, 0]
            else:
                column_axes = axes[:, col_idx]
            
            # Create heatmaps for each column (rows correspond to different metrics)
            for row_idx, col_name in enumerate(col_names):
                # Get the axis for this specific subplot
                current_ax = axes[row_idx, col_idx]
                
                # Call plot_flexible_heatmap for this column with grid_by=None
                _, _ = plot_flexible_heatmap(
                    df=grid_df,
                    variable_config=variable_config,  # Already removed grid_by
                    fixed_values=temp_fixed_values,
                    value_columns=col_name,
                    cmaps=self.blue_to_white_to_red,
                    annot=False,
                    reverse_x_axis=False,
                    reverse_y_axis=True,
                    colorbar_location=None,  # No colorbar
                    figsize=None,
                    axis_label_fontsize=self.axis_label_fontsize-15,
                    tick_fontsize=self.tick_fontsize,
                    vmins=[0],
                    vmaxs=[1],
                    # Only set labels on relevant edges
                    x_axis_label="",
                    y_axis_label="",
                    ax=current_ax
                )

                if row_idx != len(col_names) - 1:
                    current_ax.set_xlabel("")
                    current_ax.set_xticklabels([])

                # increse padding on x axis label
                if row_idx == len(col_names) - 1:
                    current_ax.set_xlabel(self.var_to_title["checkpoint"], fontsize=self.axis_label_fontsize-8, labelpad=25)

                # Set titles for the top row
                if row_idx == 0 and grid_by_var is not None:
                    if grid_by_var == "mlp_expansion_factor" and "hidden_size" in config:
                        value_display = int(grid_value * config["hidden_size"])
                        title = f"{self.var_to_title[grid_by_var]} = {value_display}"
                    else:
                        title = f"{self.var_to_title.get(grid_by_var, grid_by_var)} = {grid_value}"
                    
                    current_ax.set_title(title, fontsize=self.sideplot_title_fontsize)
                
                # Add row titles if provided
                if col_idx == 0 and row_titles[row_idx]:
                    current_ax.text(-0.2, 0.5, row_titles[row_idx],
                                   va='center', ha='center',
                                   rotation=90,
                                   transform=current_ax.transAxes,
                                   fontsize=self.axis_label_fontsize)
                
                # Format x-axis ticks to show every 3rd label (only for bottom row)
                if row_idx == num_rows - 1:
                    format_axis_labels_in_thousands(current_ax, axis="x")
                    xticks = current_ax.get_xticklabels()
                    for i, tick in enumerate(xticks):
                        if i % 3 != 0 or i == len(xticks) - 1:  # Hide every 3rd label and the last label
                            tick.set_visible(False)
                # Remove y-tick labels for all columns except the first
                if col_idx > 0:
                    current_ax.set_ylabel("")
                    current_ax.set_yticklabels([])

        if num_rows == 1:
            # set y label for the only row
            axes[0, 0].set_ylabel(self.var_to_title["num_tasks"], fontsize=self.axis_label_fontsize, labelpad=25)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.1)
        
        if save:
            col_str = "_".join([col.replace("_", "-") for col in col_names])
            if config.get("custom_name", None) is not None:
                filepath = self.format_plot_name(self.setting, fixed_values, config["custom_name"])
            else:
                filepath = self.format_plot_name(self.setting, fixed_values, f"titration_{col_str}_{mode}")
            self.save_figure(fig, filepath)
        
        if show:
            plt.show()
            plt.close(fig)

    def generate_free_parameter_titration_plot(self, config, show=False, save=True):
        """
        Generate a titration plot with multiple rows, each corresponding to a different metric.
        
        Args:
            config (dict): Configuration dictionary with the following keys:
                param_name: The name of the parameter to titrate
                grid_by: The variable to grid by
                hidden_size: The hidden size
                custom_name: The custom name of the plot (if not provided, the default name will be used)
            show (bool): Whether to display the plot
            save (bool): Whether to save the plot
        """
        grid_by = config["grid_by"]
        if grid_by == "checkpoint" or grid_by == "num_tasks":
            raise ValueError("grid_by cannot be checkpoint or num_tasks")
        
        unique_grid_by_values = self.transformer_df[grid_by].unique()
        if grid_by == "mlp_expansion_factor":
            grid_by_values_to_plot = [x*config["hidden_size"] for x in unique_grid_by_values]
        else:
            grid_by_values_to_plot = unique_grid_by_values

        # set colors based on which grid_by value is being iterated over
        if grid_by == "mlp_expansion_factor":
            colors = [self.blue_to_white_to_red(x*0.05) for x in range(len(unique_grid_by_values))]  # Dark to light blue
        else:
            colors = [self.blue_to_white_to_red(x*0.1+0.55) for x in range(len(unique_grid_by_values))]  # Light to dark red

        param_name = config["param_name"]

        # make grid with two plots on top of each other
        fig, axes = plt.subplots(2, 1, figsize=(8, 14))

        # First plot: Parameter effects across MLP widths
        ax1 = axes[0]

        # find unique parameter values
        unique_param_values = np.exp((self.transformer_df.apply(lambda x: x["bms_results"]["params"][f"log_{param_name}"], axis=1).unique()))
        if param_name == "alpha":
            unique_param_values = 1-unique_param_values # convert to 1-alpha to get likelihood exponent

        # plot unique parameter values
        ax1.plot(grid_by_values_to_plot, unique_param_values, 'o', color='black', linewidth=2, markersize=6)

        # Add regression line
        slope, intercept = np.polyfit(np.log10(grid_by_values_to_plot), np.log10(unique_param_values), 1)
        possible_mlp_values = np.linspace(np.log10(grid_by_values_to_plot[0]), np.log10(grid_by_values_to_plot[-1]), 100)
        regression_line = 10**(slope * possible_mlp_values + intercept)
        ax1.plot(10**possible_mlp_values, regression_line, '--', color='black', linewidth=1, alpha=0.7)

        # Format axis labels
        ax1.set_xlabel(self.var_to_title[grid_by], fontsize=self.axis_label_fontsize)
        ax1.set_ylabel(f"{self.var_to_title[param_name]}", fontsize=self.axis_label_fontsize)

        # set tick fontsize
        ax1.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)


        # second plot: parameter effects across grid_by values
        ax2 = axes[1]

        # Determine what to plot based on param_name
        if param_name == 'beta':
            y_column = 'log_prior_odds_id'
            y_label = '$\\Delta K(D)^{\\beta}$'
            x_axis_var = 'num_tasks'
            x_label = self.var_to_title["num_tasks"]
            scale = 'log'
            scale_base = 2
        else:  # alpha or gamma
            y_column = 'log_bayes_factor_id'
            y_label = f'$\\Delta L(D)$' 
            x_axis_var = 'checkpoint'  # Training steps N
            x_label = self.var_to_title["checkpoint"]
            scale = 'log'
            scale_base = 10

        for i, grid_by_value in enumerate(unique_grid_by_values):
            # filter df for this grid_by value
            df = self.transformer_df.query(f"{grid_by} == {grid_by_value}", engine="python")

            # Plot the data
            if y_column == 'log_prior_odds_id':
                y_data = df.apply(lambda x: -1*x["bms_results"][y_column], axis=1)
            elif y_column == 'log_bayes_factor_id':
                y_data = df.apply(lambda x: x["bms_results"][y_column], axis=1) 
            
            ax2.plot(df[x_axis_var], y_data,
                    color=colors[i],
                    marker="o",
                    markersize=8,
                    alpha=0.7,
                    markeredgecolor='black',
                    markeredgewidth=0.25,
                    linewidth=2.5)

        # set x axis to log scale
        ax2.set_xscale(scale, base=scale_base)

        # set y axis to log scale
        ax2.set_yscale(scale, base=scale_base)
        
        # set axis labels
        ax2.set_xlabel(x_label, fontsize=self.axis_label_fontsize)
        ax2.set_ylabel(y_label, fontsize=self.axis_label_fontsize)
        
        # set tick fontsize
        ax2.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)

        # add vertical space between plots
        plt.subplots_adjust(hspace=0.25)

        plt.tight_layout()


        if save:
            config["grid_by"] = {config["grid_by"]: "grid_by"}
            if config.get("custom_name", None) is not None:
                filepath = self.format_plot_name(self.setting, config["grid_by"], config["custom_name"])
            else:
                filepath = self.format_plot_name(self.setting, config["grid_by"], f"free_parameter_titration_{param_name}")
            self.save_figure(fig, filepath)
        
        if show:
            plt.show()
            plt.close(fig)

    def generate_loss_titration_plot(self, config, show=False, save=True):
        """
        Plot delta loss (memorized_id_nll - generalized_id_nll) as a function of a variable.
        
        Args:
            config (dict): A dictionary of configuration parameters, with the following keys:
                variable: The variable to plot against ('context_length' or 'num_tasks')
                fixed_values: Dictionary of fixed parameter values (should not include the variable being plotted)
                custom_name: The custom name of the plot (if not provided, the default name will be used)
            show (bool): Whether to display the plot
            save (bool): Whether to save the plot
        """
        variable = config["variable"]
        fixed_values = config["fixed_values"]
                
        # Create variable config for plotting
        fixed_vars = list(fixed_values.keys())
        variable_config = {
            "fixed": fixed_vars,
            "iterate_over": variable,
            "grid_by": None,
        }
        
        # Create the plot
        fig, ax = plot_flexible_scatterplot(
            df=self.algo_df,
            variable_config=variable_config,
            fixed_values=fixed_values,
            value_column=[lambda x: x["generalized_id_nll"] - x["memorized_id_nll"]],
            needed_cols_for_callable=["memorized_id_nll", "generalized_id_nll"],
            x_scale="log" if variable == "num_tasks" else "linear",
            x_scale_base=2 if variable == "num_tasks" else 10,
            y_label="$\\Delta L$",
            x_label=self.var_to_title[variable],
            color=["black"],
            linewidth=3,
            linestyle=["-"],
            marker=["o"],
            markersize=8,
            legend_labels=None,
            show_title=False,
            axis_label_fontsize=self.axis_label_fontsize,
            tick_fontsize=self.tick_fontsize,
            figsize=(8, 6)
        )
        
        # Remove legend if it exists (since we only have one line)
        if ax.legend_:
            ax.legend_.remove()        
        if save:
            if config.get("custom_name", None) is not None:
                filepath = self.format_plot_name(self.setting, fixed_values, config["custom_name"])
            else:
                filepath = self.format_plot_name(self.setting, fixed_values, f"loss_titration_{variable}")
            self.save_figure(fig, filepath)
        
        if show:
            plt.show()
            plt.close(fig)

    def generate_relative_and_absolute_distance_plot(self, config, show=False, save=True):
        """
        Generate a plot showing both relative and absolute distances across different task diversities.
        
        Args:
            config (dict): A dictionary of configuration parameters, with the following keys:
                fixed_values: {context_length, mlp_expansion_factor, num_dims}
                mode (optional): The evaluation mode (train, eval)
                custom_name: The custom name of the plot (if not provided, the default name will be used)
            show (bool): Whether to display the plot
            save (bool): Whether to save the plot
        """
        fixed_values = config["fixed_values"]
        mode = config.get("mode", "train")
        
        # Filter DataFrame based on fixed values
        filtered_df = self.transformer_df.query(
            "context_length == @fixed_values['context_length'] & "
            "mlp_expansion_factor == @fixed_values['mlp_expansion_factor'] & "
            "num_dims == @fixed_values['num_dims']", engine="python"
        )
        
        # Get unique task diversity values and sort them in DESCENDING order (highest at top)
        num_tasks_values = sorted(filtered_df['num_tasks'].unique(), reverse=True)
        selected_tasks = num_tasks_values
        num_rows = len(selected_tasks)
        
        # Create figure with subplots - more vertical spacing between plots
        fig, axes = plt.subplots(num_rows, 1, figsize=(9, 10), sharex=True, 
                                gridspec_kw={'hspace': 0.25})  # Increased spacing between subplots
        
        if num_rows == 1:
            axes = [axes]  # Ensure axes is always a list
        
        # Find first valid checkpoint common to all task diversities
        min_valid_checkpoint = 0
        for num_tasks in selected_tasks:
            task_data = filtered_df[filtered_df['num_tasks'] == num_tasks].sort_values('checkpoint')
            if len(task_data) > 0:
                first_checkpoint = task_data['checkpoint'].min()
                min_valid_checkpoint = max(min_valid_checkpoint, first_checkpoint)
        
        # Get distance column names
        mem_distance_col = f"distance_from_memorized_{mode}"
        gen_distance_col = f"distance_from_generalized_{mode}"
        
        # Get metric name for the plot
        metric_name = self.get_metric_name(distance=True)
        
        # Use the custom colormap
        cmap = self.blue_to_white_to_red
        
        # Plot each task diversity as a separate subplot
        for i, num_tasks in enumerate(selected_tasks):
            ax = axes[i]
            
            # Filter data for this task diversity
            task_data = filtered_df[filtered_df['num_tasks'] == num_tasks].sort_values('checkpoint')
            
            if len(task_data) < 2:  # Skip if not enough data points
                continue
            
            # Get the data
            checkpoints = task_data['checkpoint'].values
            # Only use checkpoints at or after the min_valid_checkpoint
            valid_indices = checkpoints >= min_valid_checkpoint
            checkpoints = checkpoints[valid_indices]
            mem_distances = task_data[mem_distance_col].values[valid_indices]
            gen_distances = task_data[gen_distance_col].values[valid_indices]
            rel_distance_col = f"relative_distance_{mode}"
            rel_distances = task_data[rel_distance_col].values[valid_indices]
            
            # Determine y-axis limits specific to this subplot (considering both distance metrics)
            min_distance = min(mem_distances.min(), gen_distances.min())
            max_distance = max(mem_distances.max(), gen_distances.max())
            range_distance = max_distance - min_distance
            # Add padding to y-axis (10% of range)
            y_padding = range_distance * 0.1 if range_distance > 0 else 0.001
            y_min = max(0, min_distance - y_padding)
            y_max = max_distance + y_padding
            
            # Create a more detailed grid for smoother visualization
            x_points = np.linspace(checkpoints.min(), checkpoints.max(), 100)
            y_points = np.linspace(y_min, y_max, 50)
            X, Y = np.meshgrid(x_points, y_points)
            
            # Create interpolated relative distance values
            if len(checkpoints) > 1:  # Only interpolate if we have multiple points
                # Create interpolation function (use 'nearest' for values outside the range)
                interp_func = interp1d(checkpoints, rel_distances, kind='linear', 
                                       bounds_error=False, fill_value=(rel_distances[0], rel_distances[-1]))
                Z = np.zeros_like(X)
                for j in range(Z.shape[1]):
                    # Apply interpolated relative distance value for each x point
                    Z[:, j] = interp_func(X[0, j])
            else:
                # If only one point, use constant value
                Z = np.full_like(X, rel_distances[0])
            
            # Plot background color using pcolormesh
            im = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=0, vmax=1, 
                               alpha=0.7, shading='auto', zorder=1)
            
            # Plot memorized algorithm distance line (solid)
            ax.plot(checkpoints, mem_distances, color='black', linewidth=2.0, zorder=3)
            
            # Plot generalized algorithm distance line (dashed)
            ax.plot(checkpoints, gen_distances, color='black', linewidth=2.0, 
                    linestyle='--', zorder=3)
            
            # Set y-axis limits specific to this subplot
            ax.set_ylim(y_min, y_max)
            
            # Set only two y-ticks (min and max) 
            ax.set_yticks([y_min, y_max])
            ax.set_yticklabels([f"{y_min:.2f}", f"{y_max:.2f}"])
            
            # Add D value label to every second plot on the right side
            if i % 2 == 0:  # Every second row
                # Position the text to the right of the plot
                ax.text(1.02, 0.5, f"$D$ = {num_tasks}", transform=ax.transAxes, 
                       fontsize=self.tick_fontsize, va='center', ha='left')
            
            # Configure y-axis ticks
            ax.tick_params(axis='y', labelsize=self.tick_fontsize - 12)
            
            # Clean up styling - make it more subtle
            ax.grid(True, linestyle='--', alpha=0.2, zorder=2)
            
            # Remove all spines except the left one
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            
            # Remove x-tick marks from all but the bottom plot
            if i < num_rows - 1:
                ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        
        # Configure bottom plot x-axis - show x-axis values
        axes[-1].set_xlabel("Training Steps $N$", fontsize=self.axis_label_fontsize)
        axes[-1].spines['bottom'].set_visible(True)  # Show bottom spine only for the last subplot
        if config.get("xscale", "log") == "log":
            axes[-1].set_xscale('log', base=10)
        else:
            axes[-1].set_xscale('linear')
        
        # Ensure x-axis ticks and labels are visible on the bottom plot
        axes[-1].tick_params(axis='x', which='both', bottom=True, labelbottom=True, 
                            labelsize=self.tick_fontsize - 2)
        
        # Set x-axis limits to actual data range
        max_checkpoint = filtered_df['checkpoint'].max()
        for ax in axes:
            ax.set_xlim(min_valid_checkpoint, max_checkpoint)
                
        # Add a left-side label for the metric
        fig.text(0.01, 0.5, metric_name, fontsize=self.axis_label_fontsize, 
                 rotation=90, va='center', ha='center')
        
        # Adjust layout - make it tight but leave room for labels and bottom axis
        plt.tight_layout(rect=[0.07, 0.05, 0.9, 0.98])
        
        if save:
            if config.get("custom_name", None) is not None:
                filepath = self.format_plot_name(self.setting, fixed_values, config["custom_name"])
            else:
                filepath = self.format_plot_name(self.setting, fixed_values, f"relative_and_absolute_distance_{mode}")
            self.save_figure(fig, filepath)

        if show:
            plt.show()
            plt.close(fig)

    @staticmethod
    def assemble_pdfs_grid(pdf_paths, output_path, ncols, title=None, padding_inches=0.5):
        """
        Assemble multiple PDF figures into a grid layout using merge_transformed_page.
        Based on StackOverflow approach for better compatibility.
        """
        from pypdf import PdfReader, PdfWriter, Transformation
        import math
        import os

        # Read all PDFs and get their dimensions
        pdf_readers = []
        pdf_dimensions = []
        for path in pdf_paths:
            try:
                reader = PdfReader(path)
                if not reader.pages:
                    print(f"Warning: {path} has no pages, skipping")
                    continue
                page = reader.pages[0]
                mediabox = page.mediabox
                width = float(mediabox.width)
                height = float(mediabox.height)
                if width > 0 and height > 0:
                    pdf_readers.append(reader)
                    pdf_dimensions.append((width, height))
                else:
                    print(f"Warning: {path} has invalid dimensions ({width}x{height}), skipping")
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue
        
        if not pdf_readers:
            print("No valid PDFs could be loaded")
            return
        
        n_pdfs = len(pdf_readers)
        nrows = math.ceil(n_pdfs / ncols)
        
        # Find maximum dimensions for uniform grid cells
        max_width = max(dim[0] for dim in pdf_dimensions)
        max_height = max(dim[1] for dim in pdf_dimensions)
        
        # Calculate grid dimensions in points
        padding_pts = padding_inches * 72
        title_height_pts = 72 if title else 0
        
        # Calculate total canvas size
        total_width = ncols * max_width + (ncols + 1) * padding_pts
        total_height = nrows * max_height + (nrows + 1) * padding_pts + title_height_pts
        
        # Create the output PDF
        output = PdfWriter()
        
        # Add a blank page with the calculated dimensions
        output.add_blank_page(width=total_width, height=total_height)
        
        # Add title if provided
        if title:
            from reportlab.pdfgen import canvas
            import io
            
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=(total_width, total_height))
            c.setFont("Helvetica-Bold", 24)
            title_width = c.stringWidth(title, "Helvetica-Bold", 24)
            title_x = (total_width - title_width) / 2
            title_y = total_height - 36
            c.drawString(title_x, title_y, title)
            c.save()
            buffer.seek(0)
            
            title_reader = PdfReader(buffer)
            title_page = title_reader.pages[0]
            
            # Merge title page onto the main page
            output.pages[0].merge_transformed_page(
                title_page,
                Transformation()  # No transformation needed for title
            )
            buffer.close()
        
        # Place each PDF in the grid
        for idx, (orig_width, orig_height) in enumerate(pdf_dimensions):
            try:
                # Calculate grid position
                row = idx // ncols
                col = idx % ncols
                
                # Calculate position (bottom-left origin)
                x = col * (max_width + padding_pts) + padding_pts
                y = (nrows - row - 1) * (max_height + padding_pts) + padding_pts
                
                # Calculate scaling to fit in cell
                scale_x = max_width / orig_width
                scale_y = max_height / orig_height
                scale = min(scale_x, scale_y, 1.0)  # Don't scale up
                
                # Calculate centered position within cell
                scaled_width = orig_width * scale
                scaled_height = orig_height * scale
                center_x = x + (max_width - scaled_width) / 2
                center_y = y + (max_height - scaled_height) / 2
                
                # Get the source page
                source_reader = PdfReader(pdf_paths[idx])
                source_page = source_reader.pages[0]
                
                # Create transformation: scale first, then translate
                transformation = Transformation().scale(scale, scale).translate(center_x, center_y)
                
                # Merge the transformed page onto the output page
                output.pages[0].merge_transformed_page(source_page, transformation)
                
                print(f"Placed PDF {idx} at ({center_x:.1f}, {center_y:.1f}) with scale {scale:.2f}")
                
            except Exception as e:
                print(f"Error processing PDF {idx}: {e}")
                continue
        
        # Write the output file
        try:
            # Compress the content in-memory before writing
            print("Applying lossless compression...")
            for page in output.pages:
                page.compress_content_streams()  # Compress each page

            # Write the compressed PDF to the file
            with open(output_path, 'wb') as output_file:
                output.write(output_file)

            # Check final file size
            file_size = os.path.getsize(output_path)
            print(f"Successfully saved compressed PDF to {output_path}")
            print(f"Final file size: {file_size} bytes ({file_size / 1024 / 1024:.1f} MB)")

        except Exception as e:
            print(f"Error writing output PDF: {e}")
        finally:
            output.close()

    def generate_interpolation_threshold_plot(self, config, show=False, save=True):
        """
        Generate a plot showing interpolation distance over training steps with threshold visualization.
        
        Args:
            config: Dictionary containing plot configuration with 'fixed_values' key
            show: Whether to display the plot
            save: Whether to save the plot
            
        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        fixed_values = config['fixed_values']
        
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by num_tasks and checkpoint to see how interpolation distance changes
        grouped_df = self.transformer_df_all_checkpoints.query(
            "context_length == @fixed_values['context_length'] and mlp_expansion_factor == @fixed_values['mlp_expansion_factor'] and num_dims == @fixed_values['num_dims']", engine="python"
        ).groupby(['num_tasks', 'checkpoint'])['interpolation_distance_train'].mean().reset_index()
        
        # Pivot the data for easier plotting
        pivot_df = grouped_df.pivot(index='checkpoint', columns='num_tasks', values='interpolation_distance_train')
        
        # Plot each task diversity as a separate line
        cmap = plt.cm.get_cmap('Greys')
        for task_div in pivot_df.columns:
            ax.plot(pivot_df.index, pivot_df[task_div], marker='', 
                   label=f'Tasks: {task_div}', 
                   color=cmap(np.log(task_div)/np.log(pivot_df.columns[-1])))
        
        ax.tick_params(labelsize=15)
        ax.set_xlabel('Training Step $N$', fontsize=25)
        ax.set_ylabel(f'{self.get_metric_name()} (ID)', fontsize=25)

        if len(self.algo_names_dict) == 2:
            # Get threshold value
            threshold = self.transformer_df.query(
                "context_length == @fixed_values['context_length'] and "
                "mlp_expansion_factor == @fixed_values['mlp_expansion_factor'] and "
                "num_dims == @fixed_values['num_dims']", engine="python"
            )["approximate_interpolation_threshold"].iloc[0]
            
            # Add vertical line at threshold
            ax.axvline(x=threshold, color='black', linestyle='--', label=f'Checkpoint {threshold}')
            format_axis_labels_in_thousands(ax)
            
            # Create inset axes
            inset_ax = fig.add_axes([0.42, 0.57, 0.4, 0.35])  # [left, bottom, width, height]

            # Plot the same data in the inset, but only up to threshold
            filtered_pivot_df = pivot_df[pivot_df.index >= threshold]
            for task_div in filtered_pivot_df.columns:
                inset_ax.plot(filtered_pivot_df.index, filtered_pivot_df[task_div], marker='', 
                            markersize=3, linewidth=1, 
                            color=cmap(np.log(task_div)/np.log(filtered_pivot_df.columns[-1])))
            
            # Format the inset
            inset_ax.set_title(f'After thresholding at {threshold}', fontsize=9)
            inset_ax.tick_params(labelsize=8)
            format_axis_labels_in_thousands(inset_ax)
            
            # Update legend to include the vertical line
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles[:-1], labels=labels[:-1], title='Task Diversity $D$')  # Exclude vertical line from legend

            # format inset
            inset_ax.grid(False)
            inset_ax.spines['top'].set_visible(False)
            inset_ax.spines['right'].set_visible(False)

        
        # Remove grid and spines for clean appearance
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save the figure if requested
        if save:
            filepath = self.format_plot_name(self.setting, fixed_values, "interpolation_threshold")
            self.save_figure(fig, filepath)
        
        # Show the figure if requested
        if show:
            plt.show()
        
        return fig, ax