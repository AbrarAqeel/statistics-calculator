import time
import pandas as pd
import tkinter as tk
from tkinter import ttk


class Measures:
    """
    This class handles the formulas for the measure of central tendency,
    measure of position, and measure of dispersion
    """

    def __init__(self, sample_data, slider_value=None, df=None):
        self.sample_data = sample_data  # data set
        self.slider_value = slider_value  # positional values for decile, percentile and quartile
        self.df = df  # frequency distribution table (grouped data)

        # Calculating class width
        num_of_classes = 5
        data_range = max(self.sample_data) - min(self.sample_data)
        self.class_width = data_range // num_of_classes

    def mean(self):
        if self.df is None:  # Ungrouped data
            computed_mean = sum(self.sample_data) / len(self.sample_data)
        else:  # Grouped data
            computed_mean = self.df['FX'].sum() / self.df['Frequency (F)'].sum()

        return computed_mean

    def median(self):
        self.sample_data.sort()  # sorting in ascending order
        n = len(self.sample_data)

        if self.df is None:
            if n % 2 == 0:  # even number of samples
                computed_median = (self.sample_data[(n // 2) - 1] + self.sample_data[n // 2]) / 2
            else:  # odd number of samples
                computed_median = self.sample_data[(n - 1) // 2]
        else:
            median_position = (2 * (n + 1)) / 4  # second quartile
            rounded_value = -int(-median_position // 1)  # Round up

            # Find the closest index value that is less than or equal to rounded_value
            closest_index = self.df.index[self.df.index <= rounded_value].max()

            value = self.df.at[(closest_index - 1), 'Cumulative Frequency (CF)']
            lower_class = self.df.at[closest_index, 'Lower Class Limit']  # getting the value at [row, column]

            freq = self.df.at[closest_index, 'Frequency (F)']
            computed_median = lower_class + ((self.class_width / freq) * ((n / 2) - (value // 1)))

        return computed_median

    def mode(self):
        computed_mode = None

        if self.df is None:
            sample_count = {}
            for sample in self.sample_data:
                if sample in sample_count:
                    sample_count[sample] += 1
                else:
                    sample_count[sample] = 1

            max_frequency = max(sample_count.values())  # checking which data repeated the most
            for key, value in sample_count.items():
                if value == max_frequency:
                    computed_mode = key
        else:
            max_frequency = self.df['Frequency (F)'].max()
            max_frequency_index = self.df.index[max_frequency]  # index of frequency

            if 0 < max_frequency_index < 5:
                next_frequency = self.df.at[max_frequency_index + 1, 'Frequency (F)']
                previous_frequency = self.df.at[max_frequency_index - 1, 'Frequency (F)']
            else:
                previous_frequency = 0
                next_frequency = 0

            lower_class = self.df.at[max_frequency_index, 'Lower Class Limit']  # getting the value at [row, column]

            computed_mode = (lower_class +
                             ((max_frequency - previous_frequency) / (2 * max_frequency) - previous_frequency - next_frequency) *
                             self.class_width)

        return computed_mode

    def variance(self):
        mean_val = self.mean()

        if self.df is None:  # Ungrouped data
            squared_diff_sum = sum((x - mean_val) ** 2 for x in self.sample_data)
            computed_variance = squared_diff_sum / (len(self.sample_data) - 1)
        else:  # Grouped data
            squared_diff_sum = sum(freq * ((x - mean_val) ** 2) for x, freq in zip(self.df['Class Mark (X)'], self.df['Frequency (F)']))
            computed_variance = squared_diff_sum / (self.df['Frequency (F)'].sum() - 1)

        return computed_variance

    def std(self):
        computed_std = self.variance() ** 0.5
        return computed_std

    def percentile(self):
        if self.df is None:
            self.sample_data.sort()
            n = len(self.sample_data)
            k = int((self.slider_value / 100) * n)
            computed_percentile = self.sample_data[k - 1]
        else:
            target_cumulative_frequency = (self.slider_value / 100) * self.df['Frequency (F)'].sum()
            closest_index = self.df[self.df['Cumulative Frequency (CF)'] >= target_cumulative_frequency].index[0]
            lower_class = self.df.at[closest_index, 'Lower Class Limit']
            freq = self.df.at[closest_index, 'Frequency (F)']
            position_in_class = (target_cumulative_frequency - self.df.at[closest_index - 1, 'Cumulative Frequency (CF)']) / freq
            computed_percentile = lower_class + (position_in_class * self.class_width)

        return computed_percentile

    def quartile(self):
        if self.df is None:
            self.sample_data.sort()
            n = len(self.sample_data)
            if self.slider_value == 1:
                k = int(0.25 * n)
            elif self.slider_value == 2:
                k = int(0.5 * n)
            else:
                k = int(0.75 * n)
            computed_quartile = self.sample_data[k - 1]
        else:
            target_cumulative_frequency = (self.slider_value / 4) * self.df['Frequency (F)'].sum()
            closest_index = self.df[self.df['Cumulative Frequency (CF)'] >= target_cumulative_frequency].index[0]
            lower_class = self.df.at[closest_index, 'Lower Class Limit']
            freq = self.df.at[closest_index, 'Frequency (F)']
            position_in_class = (target_cumulative_frequency - self.df.at[closest_index - 1, 'Cumulative Frequency (CF)']) / freq
            computed_quartile = lower_class + (position_in_class * self.class_width)

        return computed_quartile

    def decile(self):
        if self.df is None:
            self.sample_data.sort()
            n = len(self.sample_data)
            k = int((self.slider_value / 10) * n)
            computed_decile = self.sample_data[k - 1]
        else:
            target_cumulative_frequency = (self.slider_value / 10) * self.df['Frequency (F)'].sum()
            closest_index = self.df[self.df['Cumulative Frequency (CF)'] >= target_cumulative_frequency].index[0]
            lower_class = self.df.at[closest_index, 'Lower Class Limit']
            freq = self.df.at[closest_index, 'Frequency (F)']
            position_in_class = (target_cumulative_frequency - self.df.at[closest_index - 1, 'Cumulative Frequency (CF)']) / freq
            computed_decile = lower_class + (position_in_class * self.class_width)

        return computed_decile


class StatisticsCalculatorApp:
    """
    This class handles the GUI for the calculator app.
    """

    def __init__(self):
        # Constants for font and bg/fg colour
        self.FONT = ('Helvetica bold', 12)  # font name and size
        self.BG = '#36454F'  # charcoal gray
        self.FG = '#FFFFFF'  # white

        # Configuring main window
        self.root = tk.Tk()  # instantiating tkinter window
        self.root.title('Statistics Calculator')  # window title
        self.root.geometry('600x310+450+200')  # window size and offset
        self.root.config(background=self.BG)  # bg color
        self.root.resizable(False, False)  # disabling window resizing in both directions

        # Setting an icon image
        icon = tk.PhotoImage(file='icon.png')  # pic used as icon for app
        self.root.iconphoto(True, icon)

        # Placeholder for entries
        self.sample_entry = None
        self.operation_combobox = None

        self.data_type_label = None
        self.calculate_button = None

        self.output_text = None
        self.output_label = None

        self.rb_value = None
        self.rb_grouped = None
        self.rb_ungrouped = None

        self.percentile_label = None
        self.percentile_slider = None
        self.percentile_value_label = None

        self.quartile_label = None
        self.quartile_slider = None
        self.quartile_value_label = None

        self.decile_label = None
        self.decile_slider = None
        self.decile_value_label = None

        # Calling widgets method
        self.widgets()

    def widgets(self):
        """
        This method creates all the widgets for the app.
        """
        # Samples label and entry
        samples_label = ttk.Label(self.root, text='Enter samples:', font=self.FONT, background=self.BG, foreground=self.FG)
        samples_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.sample_entry = ttk.Entry(self.root)
        self.sample_entry.grid(row=0, column=1, padx=5, pady=5, columnspan=3, sticky='we')

        # Operations label and combo box
        values = ['Mean', 'Median', 'Mode', 'Standard Deviation', 'Variance', 'Percentile', 'Quartile', 'Decile']

        operation_label = ttk.Label(self.root, text='Choose operation:', font=self.FONT, background=self.BG, foreground=self.FG)
        operation_label.grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.operation_combobox = ttk.Combobox(self.root, values=values, state='readonly')
        self.operation_combobox.grid(row=1, column=1, padx=5, pady=5, columnspan=3, sticky='we')
        self.operation_combobox.set('Mean')  # setting a default value

        # Slider label and slider bar
        self.percentile_label = ttk.Label(self.root, text='Percentile:', font=self.FONT, background=self.BG, foreground=self.FG)
        self.percentile_value_label = ttk.Label(self.root, text='', font=self.FONT, background=self.BG, foreground=self.FG)
        self.percentile_slider = ttk.Scale(self.root, from_=1, to=99, orient='horizontal',
                                           command=lambda value: self.show_slider_value(value, 'percentile'))

        self.quartile_label = ttk.Label(self.root, text='Quartile:', font=self.FONT, background=self.BG, foreground=self.FG)
        self.quartile_value_label = ttk.Label(self.root, text='', font=self.FONT, background=self.BG, foreground=self.FG)
        self.quartile_slider = ttk.Scale(self.root, from_=1, to=3, orient='horizontal',
                                         command=lambda value: self.show_slider_value(value, 'quartile'))

        self.decile_label = ttk.Label(self.root, text='Decile:', font=self.FONT, background=self.BG, foreground=self.FG)
        self.decile_value_label = ttk.Label(self.root, text='', font=self.FONT, background=self.BG, foreground=self.FG)
        self.decile_slider = ttk.Scale(self.root, from_=1, to=9, orient='horizontal',
                                       command=lambda value: self.show_slider_value(value, 'decile'))

        # Data type label and radio buttons
        self.data_type_label = ttk.Label(self.root, text='Data type:', font=self.FONT, background=self.BG, foreground=self.FG)

        self.rb_value = tk.IntVar()  # to hold integer data
        self.rb_value.set(1)  # setting a default value

        self.rb_grouped = ttk.Radiobutton(self.root, text='Grouped', value=1, variable=self.rb_value)
        self.rb_ungrouped = ttk.Radiobutton(self.root, text='Ungrouped', value=2, variable=self.rb_value)

        # Calculate button
        self.calculate_button = ttk.Button(self.root, text='Calculate', command=self.get_data)

        # Output label
        self.output_label = ttk.Label(self.root, text='Output:', font=self.FONT, background=self.BG, foreground=self.FG)

        # Column configure
        self.root.columnconfigure(2, weight=1)

        # Calling other_widgets method
        self.other_widgets()

        # Combobox event binding
        self.operation_combobox.bind('<<ComboboxSelected>>', lambda event: self.update_interface())

        # Bind <Escape> key press event to terminate the program
        self.root.bind('<Escape>', lambda event: self.root.quit())

    def update_interface(self):
        """
        This method updates the interface based on the selected operation.
        """
        # Getting operation value
        operation = self.operation_combobox.get()

        # Calling hide_other_widgets method
        self.hide_other_widgets()

        if operation in ['Percentile', 'Quartile', 'Decile']:
            self.slider_widgets(operation)
        else:
            self.hide_slider_widgets('Percentile', 'Quartile', 'Decile')
            self.other_widgets()

    def slider_widgets(self, operation):
        """
        This method will show the slider widget when its label is selected
        """
        if operation == 'Percentile':
            # Show Percentile label, slider, and value label, and hide the others
            self.percentile_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
            self.percentile_slider.grid(row=2, column=1, padx=5, pady=5, columnspan=3, sticky='we')
            self.percentile_value_label.grid(row=1, column=1, padx=1, pady=5, sticky='w')
            self.hide_slider_widgets('Quartile', 'Decile')
        elif operation == 'Quartile':
            # Show Quartile label, slider, and value label, and hide the others
            self.quartile_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
            self.quartile_slider.grid(row=2, column=1, padx=5, pady=5, columnspan=3, sticky='we')
            self.quartile_value_label.grid(row=1, column=1, padx=1, pady=5, sticky='w')
            self.hide_slider_widgets('Percentile', 'Decile')
        else:
            # Show Decile label, slider, and value label, and hide the others
            self.decile_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
            self.decile_slider.grid(row=2, column=1, padx=5, pady=5, columnspan=3, sticky='we')
            self.decile_value_label.grid(row=1, column=1, padx=1, pady=5, sticky='w')
            self.hide_slider_widgets('Percentile', 'Quartile')

        # Calling other_widgets method
        self.other_widgets(3, 4, 5, 8)

    def hide_slider_widgets(self, *operations):
        """
        This method hides certain widgets based on the given operations.

        :param operations: One or more operations to hide the corresponding widgets.
        """
        if 'Percentile' in operations:
            self.percentile_label.grid_forget()
            self.percentile_slider.grid_forget()
            self.percentile_value_label.grid_forget()

        if 'Quartile' in operations:
            self.quartile_label.grid_forget()
            self.quartile_slider.grid_forget()
            self.quartile_value_label.grid_forget()

        if 'Decile' in operations:
            self.decile_label.grid_forget()
            self.decile_slider.grid_forget()
            self.decile_value_label.grid_forget()

    def show_slider_value(self, value, slider_type):
        """
        This method displays the value of the slider temporarily above the slider.

        :param value: The current value of the slider.
        :param slider_type: The type of the slider (percentile, quartile, or decile).
        """
        if slider_type == 'percentile':
            self.percentile_value_label.config(text=value)
            self.root.after(1500, lambda: self.percentile_value_label.config(text=''))
        elif slider_type == 'quartile':
            self.quartile_value_label.config(text=value)
            self.root.after(1500, lambda: self.quartile_value_label.config(text=''))
        elif slider_type == 'decile':
            self.decile_value_label.config(text=value)
            self.root.after(1500, lambda: self.decile_value_label.config(text=''))

    def other_widgets(self, r1=2, r2=3, r3=4, h=10):
        """
        This method will show the other widgets based on the position of slider widgets.

        :param r1: Default row position of '2' unless specified.
        :param r2: Default row position of '3' unless specified.
        :param r3: Default row position of '4' unless specified.
        :param h: Default text box height of '10' unless specified.
        """
        # Data type label and radio buttons position
        self.data_type_label.grid(row=r1, column=0, padx=5, pady=5, sticky='w')

        self.rb_grouped.grid(row=r1, column=1, padx=5, pady=5, sticky='w')
        self.rb_ungrouped.grid(row=r1, column=2, padx=5, pady=5, sticky='w')

        # Calculate button position
        self.calculate_button.grid(row=r1, column=3, padx=5, pady=5, sticky='e')

        # Output label and text box positions
        self.output_label.grid(row=r2, column=0, padx=5, pady=5, sticky='w')
        self.output_text = tk.Text(self.root, wrap=tk.WORD, height=h)
        self.output_text.grid(row=r3, column=0, padx=5, pady=5, columnspan=4, sticky='we')

    def hide_other_widgets(self):
        """
        This method will hide the widgets created by other_widgets method
        """
        # Hide data type label and radio buttons position
        self.data_type_label.grid_forget()

        self.rb_grouped.grid_forget()
        self.rb_ungrouped.grid_forget()

        # Hide calculate button position
        self.calculate_button.grid_forget()

        # Hide output label and text box positions
        self.output_label.grid_forget()
        self.output_text.grid_forget()

    def get_data(self):
        """
        This method will collect data such as entered samples, operation and data type
        and pass it to another method (which will compute the data) then will show
        the results and the time taken to compute data as output.
        """
        try:
            sample_data = list(map(float, self.sample_entry.get().split(',')))
        except ValueError as e:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Error: {e} \n")
            self.output_text.insert(tk.END, "Please enter integer or float data only!")
        else:
            operation = self.operation_combobox.get()

            if operation == 'Percentile':
                slider_value = self.percentile_slider.get()
            elif operation == 'Quartile':
                slider_value = self.quartile_slider.get()
            elif operation == 'Decile':
                slider_value = self.decile_slider.get()
            else:
                slider_value = None

            data_type = self.rb_value.get()

            if data_type == 1:
                method = 'Grouped'
            else:
                method = 'Ungrouped'

            results, elapsed_time = self.compute_data(sample_data, operation, data_type, slider_value)

            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"By {method} method: \n")

            if operation not in ('Percentile', 'Quartile', 'Decile'):
                self.output_text.insert(tk.END, f"{operation}: {results:.2f} \n")
            else:
                self.output_text.insert(tk.END, f"{operation}({slider_value}): {results:.2f} \n")

            self.output_text.insert(tk.END, f"Time taken: {elapsed_time:.6f} seconds \n")

    def compute_data(self, sample_data, operation, data_type, slider_value):
        """
        This method computes the data gathered.

        :param sample_data: Total samples user have entered.
        :param operation: Value from operation combobox.
        :param data_type: Grouped or Ungrouped method.
        :param slider_value: Percentile, quartile or decile slider value.

        :return: results and time taken to compute data.
        """
        start_time = time.time()

        # Convert sample_data to a DataFrame if data_type is 'Grouped'
        if data_type == 1:
            df = self.create_dataframe(sample_data)
        else:
            df = None

        # Create an instance of the Measures class
        measures = Measures(sample_data, slider_value, df)

        # Perform the appropriate calculation based on the selected operation
        if operation == 'Mean':
            result = measures.mean()
        elif operation == 'Median':
            result = measures.median()
        elif operation == 'Mode':
            result = measures.mode()
        elif operation == 'Standard Deviation':
            result = measures.std()
        elif operation == 'Variance':
            result = measures.variance()
        elif operation == 'Percentile':
            result = measures.percentile()
        elif operation == 'Quartile':
            result = measures.quartile()
        elif operation == 'Decile':
            result = measures.decile()
        else:
            result = None

        elapsed_time = time.time() - start_time

        return result, elapsed_time

    @staticmethod
    def create_dataframe(sample_data):
        """
        This method creates a DataFrame for the sample data.

        :param sample_data: The sample data provided by the user.

        :return: The DataFrame representing the sample data.
        """
        # Placeholders
        lower_class_limit = []
        upper_class_limit = []
        lower_class_boundary = []
        upper_class_boundary = []
        class_mark = []
        frequency = []
        cumulative_frequency = []
        relative_frequency = []
        cumulative_relative_frequency = []
        fx = []
        sq_mean_mid_diff = []

        # Calculating class interval
        num_of_classes = 5
        data_range = max(sample_data) - min(sample_data)
        class_width = data_range // num_of_classes

        for i in range(num_of_classes):
            # Class limit
            lower_limit = min(sample_data) + (i * class_width)
            upper_limit = lower_limit + class_width
            lower_class_limit.append(lower_limit)
            upper_class_limit.append(upper_limit)

            # Class boundary
            remainder = (upper_class_limit[i] - lower_class_limit[i]) % 2
            lower_class_boundary.append(lower_class_limit[i] - remainder)
            upper_class_boundary.append(upper_class_limit[i] + remainder)

            # Class mark (x)
            class_mid_point = (lower_class_limit[i] + upper_class_limit[i]) / 2
            class_mark.append(class_mid_point)

            # Frequency
            freq = 0
            for sample in sample_data:
                if lower_class_limit[i] <= sample <= upper_class_limit[i]:
                    freq += 1
            frequency.append(freq)

        # Cumulative and relative frequency
        cf = 0
        crf = 0
        for f in frequency:
            # Cumulative frequency
            cf += f
            cumulative_frequency.append(cf)
            # Relative frequency
            r_freq = f / len(sample_data)
            relative_frequency.append(r_freq)

        # Cumulative relative frequency
        for rf in relative_frequency:
            crf += rf
            cumulative_relative_frequency.append(crf)

        # FX
        for i in range(num_of_classes):
            fx.append(frequency[i] * class_mark[i])

        # Squared difference
        mean = sum(fx) / sum(frequency)
        for i in range(num_of_classes):
            mean_mid_diff = frequency[i] * ((class_mark[i] - mean) ** 2)
            sq_mean_mid_diff.append(mean_mid_diff)

        # Making DataFrame
        data = {'Lower Class Limit': lower_class_limit,
                'Upper Class Limit': upper_class_limit,
                'Lower Class Boundary': lower_class_boundary,
                'Upper Class Boundary': upper_class_boundary,
                'Class Mark (X)': class_mark,
                'Frequency (F)': frequency,
                'Cumulative Frequency (CF)': cumulative_frequency,
                'Relative Frequency (RF)': relative_frequency,
                'Cumulative Relative Frequency (CRF)': cumulative_relative_frequency,
                'FX': fx,
                'Squared Difference': sq_mean_mid_diff}

        df = pd.DataFrame(data)

        return df

    def run(self):
        """
        This method calls the method to run the whole program.
        """
        self.root.mainloop()


if __name__ == '__main__':
    app = StatisticsCalculatorApp()
    app.run()
