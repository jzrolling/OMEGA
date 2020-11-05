import configparser, os
from ipywidgets.widgets import Layout, FloatSlider, FloatRangeSlider, IntRangeSlider, Checkbox
from ipywidgets.widgets import IntSlider, ToggleButton, Button, Text, Dropdown, Box, VBox, HBox, Tab
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from ipywidgets import GridspecLayout
import time
from ipyfilechooser import FileChooser
import OMEGA as om
import tifffile


class Image_configurations_interactive:

    def __init__(self):

        self.configurated = False
        self.config = None

        #################################. parameter sliders #############################################
        slider_layout = Layout(width='248px')
        self.crop_edge = FloatSlider(value=0.0, min=0.0, max=0.4, step=0.02, description='',
                                     readout_format='.2f', layout=slider_layout)

        self.area_pixel_range = IntRangeSlider(value=[40, 3000], min=10, max=10000, step=10,
                                               description='', readout_format='.0d', layout=slider_layout)

        self.clump_threshold = FloatSlider(value=2.2, min=2, max=3, step=0.1, description='',
                                           readout_format='.1f', layout=slider_layout)

        self.clump_sig = IntSlider(value=10, min=1, max=20, step=1, description='',
                                   readout_format='.0d', layout=slider_layout)

        self.set_button = ToggleButton(value=False, description="Set",
                                       layout=Layout(width='248px'), button_style='success')

        self.reset_button = ToggleButton(value=False, description="Reset",
                                         layout=Layout(width='248px'), button_style='warning')

        self.layer1 = HBox([Button(description='Discarded edge width', layout=Layout(width='248px')),
                            Button(description='Particle size range [px]', layout=Layout(width='248px'))])

        self.layer2 = HBox([self.crop_edge, self.area_pixel_range])

        self.layer3 = HBox([Button(description='Remove clump threshold', layout=Layout(width='248px')),
                            Button(description='Remove clump radius [px]', layout=Layout(width='248px'))])

        self.layer4 = HBox([self.clump_threshold, self.clump_sig])

        self.layer5 = HBox([self.reset_button, self.set_button])

        self.primary_display = VBox([self.layer1, self.layer2, self.layer3, self.layer4, self.layer5])

        self.reset_button.observe(self.reset_handler)
        self.set_button.observe(self.set_handler)

    def reset_handler(self, b):
        self.crop_edge.disabled = False
        self.area_pixel_range.disabled = False
        self.clump_threshold.disabled = False
        self.clump_sig.disabled = False
        self.set_default_config_data()

    def set_handler(self, b):
        # currenttime = time.ctime()
        # currenttime = '_'.join(currenttime.split(" "))
        self.config["image"]['crop_edge'] = str(self.crop_edge.value)
        self.config['image']['area_pixel_low'] = str(self.area_pixel_range.value[0])
        self.config['image']['area_pixel_high'] = str(self.area_pixel_range.value[0])
        self.config['image']['clump_threshold'] = str(self.clump_threshold.value)
        self.config['image']['clump_sigma'] = str(self.clump_sig.value)
        self.crop_edge.disabled = True
        self.area_pixel_range.disabled = True
        self.clump_threshold.disabled = True
        self.clump_sig.disabled = True
        print("Current configurations saved!")
        # with open('configuration_{}.ini'.format(currenttime), 'w') as configfile:
        # self.config.write(configfile)

    def read_default_config_data(self, configfile):
        self.config = configparser.ConfigParser()
        if configfile == None:
            self.config.read("OMEGA/configuration.ini")
        else:
            self.config.read(configfile)

    def set_default_config_data(self):
        if self.config == None:
            print('No configuration found!')
        else:
            self.crop_edge.value = float(self.config["DEFAULT"]['crop_edge'])
            self.area_pixel_range.value = [int(self.config['DEFAULT']['area_pixel_low']), \
                                           int(self.config['DEFAULT']['area_pixel_high'])]
            self.clump_threshold.value = float(self.config['DEFAULT']['clump_threshold'])
            self.clump_sig.value = float(self.config['DEFAULT']['clump_sigma'])


class Cluster_configurations_interactive:

    def __init__(self):

        self.config = None
        slider_layout = Layout(width='248px')
        #################################. parameter sliders #############################################
        self.shapeindex_threshold = IntRangeSlider(value=[0, 255], min=0, max=255, step=1,
                                                   description='', readout_format='.0d',
                                                   layout=slider_layout)

        self.seed_size = IntSlider(value=50, min=0, max=1000, step=10, description='',
                                   readout_format='.0d', layout=slider_layout)

        self.length_range = IntRangeSlider(value=[0, 50], min=0, max=50, step=1, description='',
                                           readout_format='.0d', layout=slider_layout)

        self.width_range = FloatRangeSlider(value=[0, 5], min=0, max=5, step=0.1, description='',
                                            readout_format='.1f', layout=slider_layout)

        self.deflection_ang = FloatSlider(value=0, min=0, max=1, step=0.01, description='',
                                          readout_format='.2f', layout=slider_layout)

        self.intensity_deviation = FloatSlider(value=1, min=0.1, max=5, step=0.1, description='',
                                               readout_format='.1f', layout=slider_layout)

        self.set_button = ToggleButton(value=False, description="Set",
                                       layout=Layout(width='248px'), button_style='success')

        self.reset_button = ToggleButton(value=False, description="Reset",
                                         layout=Layout(width='248px'), button_style='warning')

        self.layer1 = HBox([Button(description='Shapeindex threshold', layout=Layout(width='248px')),
                            Button(description='Watershed seed size [px]', layout=Layout(width='248px'))])

        self.layer2 = HBox([self.shapeindex_threshold, self.seed_size])

        self.layer3 = HBox([Button(description='Cell length [um]', layout=Layout(width='248px')),
                            Button(description='Cell width [um]', layout=Layout(width='248px'))])

        self.layer4 = HBox([self.length_range, self.width_range])

        self.layer5 = HBox([Button(description='Maximum particle deflection', layout=Layout(width='248px')),
                            Button(description='Signal deviation tolerance', layout=Layout(width='248px'))])

        self.layer6 = HBox([self.deflection_ang, self.intensity_deviation])

        self.layer7 = HBox([self.reset_button, self.set_button])

        self.primary_display = VBox([self.layer1, self.layer2, self.layer3,
                                     self.layer4, self.layer5, self.layer6, self.layer7])

        self.reset_button.observe(self.reset_handler)
        self.set_button.observe(self.set_handler)

    def reset_handler(self, b):
        self.shapeindex_threshold.disabled = False
        self.seed_size.disabled = False
        self.length_range.disabled = False
        self.width_range.disabled = False
        self.deflection_ang.disabled = False
        self.intensity_deviation.disabled = False
        self.set_default_config_data()

    def set_handler(self, b):
        # currenttime = time.ctime()
        # currenttime = '_'.join(currenttime.split(" "))
        self.config["cluster"]['shape_index_low_bound'] = str(self.shapeindex_threshold.value[0])
        self.config['cluster']['shape_index_high_bound'] = str(self.shapeindex_threshold.value[1])
        self.config['cluster']['seed_size_min'] = str(self.seed_size.value)
        self.config['cluster']['length_low'] = str(self.length_range.value[0])
        self.config['cluster']['length_high'] = str(self.length_range.value[1])
        self.config['cluster']['width_low'] = str(self.width_range.value[0])
        self.config['cluster']['width_high'] = str(self.width_range.value[1])
        self.config['cluster']['max_deflection_angle'] = str(self.deflection_ang.value)
        self.config['cluster']['intensity_deviation'] = str(self.intensity_deviation.value)
        self.shapeindex_threshold.disabled = True
        self.seed_size.disabled = True
        self.length_range.disabled = True
        self.width_range.disabled = True
        self.deflection_ang.disabled = True
        self.intensity_deviation.disabled = True
        print("Apply current configurations saved!")

    def set_default_config_data(self, config):
        if config == None:
            print('No configuration found!')
        else:
            self.config = config
            self.shapeindex_threshold.value = [int(self.config['DEFAULT']['shape_index_low_bound']), \
                                               int(self.config['DEFAULT']['shape_index_high_bound'])]
            self.seed_size.value = int(self.config['DEFAULT']['seed_size_min'])
            self.length_range.value = [int(self.config['DEFAULT']['length_low']), \
                                       int(self.config['DEFAULT']['length_high'])]
            self.width_range.value = [float(self.config['DEFAULT']['width_low']), \
                                      float(self.config['DEFAULT']['width_high'])]
            self.deflection_ang.value = float(self.config['DEFAULT']['max_deflection_angle'])
            self.intensity_deviation.value = float(self.config['DEFAULT']['intensity_deviation'])


class Simple_GUI:

    def __init__(self, folder=None):
        if folder == None:
            folder = os.getcwd()
        self.fc = FileChooser(folder)
        self.fileformat = None
        self.image_filename = None
        self.image = None
        self.config_filename = None
        self.mask_channel_name = None

        #################################. File format selection .#############################################
        self.header1 = Button(description='Step 1. Choose image file type.',
                              layout=Layout(width='500px'), button_style='info')

        default_option = [True, False, False]

        self.nd2b = Checkbox(value=True, description='*.nd2', disabled=False, indent=False,
                             layout=Layout(width='80px'))

        self.tiffstack = Checkbox(value=False, description='*.tiff, stacked', disabled=False,
                                  indent=False, layout=Layout(width='120px'))

        self.tiff = Checkbox(value=False, description='*.tiff (separate tiff in a folder)',
                             disabled=False, indent=False, layout=Layout(width='205px'))

        self.confirm_button = Button(description='Set', layout=Layout(width='80px'))

        self.checkbox_bbox = Box([self.nd2b, self.tiffstack, self.tiff, self.confirm_button],
                                 layout=Layout(width='503px'))

        self.step1box = VBox([self.header1, self.checkbox_bbox])

        #################################. Select file from folder .###########################################
        self.header2 = Button(description='Step 2. Select image file(s).', layout=Layout(width='500px'),
                              button_style='info')

        self.step12box = VBox([self.header1, self.checkbox_bbox, self.header2, self.fc])

        ####################################. OMEGA data import .##############################################
        self.header3 = Button(description='Step 3. Select segmentation mask image.',
                              layout=Layout(width='500px'), button_style='info')

        self.import_button = Button(description='Import', layout=Layout(width='80px'))

        self.setmask_button = Button(description='Set as mask', layout=Layout(width='110px'))

        self.channel_dropdown = Dropdown(options=['None'], layout=Layout(width='240'))

        self.channelbbox = HBox([self.import_button, self.channel_dropdown, self.setmask_button])

        ####################################. Bind part I .####################################################
        self.part_I_box = VBox([self.header1, self.checkbox_bbox, self.header2, self.fc,
                                self.header3, self.channelbbox],
                               layout=Layout(width='510px'))

        ###############################. Select configuration file .###########################################
        self.header4 = Button(description='Step 4. Choose default configurations',
                              layout=Layout(width='500px'), button_style='info')

        self.fc2 = FileChooser(folder)

        self.fc2._select.on_click(self.config_selected)

        ###############################. Image configuration panel .###########################################
        self.header5 = Button(description='Step 5. Adjust image configurations',
                              layout=Layout(width='500px'), button_style='info')

        self.configurator = Image_configurations_interactive()
        self.init_segmentation_button = Button(description='Run', layout=Layout(width='96px'))
        self.segmentation_dropdown = Dropdown(options=['None'], layout=Layout(width='240'))
        self.record_button = Button(description='Record', layout=Layout(width='96px'))
        self.preview_layer = HBox([self.init_segmentation_button,
                                   self.segmentation_dropdown,
                                   self.record_button])

        self.part_II_box = VBox([self.header4, self.fc2, self.configurator.primary_display,
                                 self.preview_layer])

        ###############################. Cluster configuration panel .##########################################
        self.header6 = Button(description='Step 6. Adjust cluster configurations',
                              layout=Layout(width='500px'), button_style='info')
        self.clusterconfig = Cluster_configurations_interactive()
        self.clustersegmentation_button = Button(description='Run', layout=Layout(width='96px'))
        self.part_III_box = VBox([self.header6, self.clusterconfig.primary_display])

        self.nd2b.observe(self.nd2handler, names='value')
        self.tiffstack.observe(self.stackhandler, names='value')
        self.tiff.observe(self.tiffhandler, names='value')
        self.confirm_button.on_click(self.fileformat_confirmed)
        self.fc._select.on_click(self.file_selected)
        self.import_button.on_click(self.import_image)
        self.channel_dropdown.observe(self.channel_dropdown_handler, names='value')
        self.setmask_button.on_click(self.setmask_handler)
        self.init_segmentation_button.on_click(self.init_segmentation_handler)
        self.segmentation_dropdown.observe(self.segmentation_dropdown_handler, names='value')
        self.record_button.on_click(self.record_configuration_handler)

        self.tabview = Tab(children=[self.part_I_box,
                                     self.part_II_box,
                                     self.part_III_box])
        self.tabview.set_title(0, 'I. Import data')
        self.tabview.set_title(1, 'II. Image CONFIG')
        self.tabview.set_title(2, 'III. Cluster CONFIG')

    def nd2handler(self, change):
        if change['new']:
            self.nd2b.value, self.tiffstack.value, self.tiff.value = True, False, False
        else:
            self.nd2b.value = False

    def stackhandler(self, change):
        if change['new']:
            self.nd2b.value, self.tiffstack.value, self.tiff.value = False, True, False
        else:
            self.tiffstack.value = False

    def tiffhandler(self, change):
        if change['new']:
            self.nd2b.value, self.tiffstack.value, self.tiff.value = False, False, True
        else:
            self.tiffstack.value = False

    def fileformat_confirmed(self, b):
        if self.nd2b.value:
            self.fileformat = 0
            self.nd2b.disabled, self.tiffstack.disabled, self.tiff.disabled = True, True, True
            print('File format: Nikon *.nd2.')
        elif self.tiff:
            self.fileformat = 1
            self.nd2b.disabled, self.tiffstack.disabled, self.tiff.disabled = True, True, True
            print('File format: ImageJ stacked *.tif.')
        elif self.tiffstack:
            self.fileformat = 2
            self.nd2b.disabled, self.tiffstack.disabled, self.tiff.disabled = True, True, True
            print('File format: ImageJ split tiff files.')
        else:
            print('No file format selected!')

    def file_selected(self, b):
        if self.fc.selected_path:
            self.inputpath = self.fc.selected_path
            self.image_filename = self.fc.selected_path + "/" + self.fc.selected_filename
            print('Congratulations! You have selected your {} as your input file.'. \
                  format(self.fc.selected_filename))

    def config_selected(self, b):
        if self.fc2.selected_path:
            self.config_path = self.fc2.selected_path
            self.config_filename = self.fc2.selected_path + "/" + self.fc2.selected_filename
            print('Congratulations! You have selected {} as your default configuration file.'. \
                  format(self.fc2.selected_filename))
            self.configurator.read_default_config_data(self.config_filename)
            self.configurator.set_default_config_data()

    def init_segmentation_handler(self, b):
        if self.configurator.config == None:
            print('Configuration file needed.')
        else:
            self.import_image(0)
            self.image.config = self.configurator.config  # ['image']
            self.image.Crop_edge()
            self.image.Enhance_brightfield()
            self.image.Enhance_fluorescence()
            self.image.Separate_clusters()
            print("Primary segmentation done.")
            self.segmentation_dropdown.options = self.image.channels + ['Binary mask',
                                                                        'Binary mask overlay',
                                                                        'Clumpy area',
                                                                        'Shape index map',
                                                                        'Crude segmentation']

    def segmentation_dropdown_handler(self, change):
        channel = change['new']
        if channel in self.image.channels:
            img = self.image.data[channel]
            cmap = 'gist_gray'
        elif channel == 'Binary mask':
            img = self.image.mask_binary
            cmap = 'gist_gray'
        elif channel == 'Binary mask overlay':
            img = self.image.mask_binary * self.image.data[self.image.mask_channel_name]
            cmap = 'gist_gray'
        elif channel == 'Clumpy area':
            img = self.image.mask_clumpy_zone
            cmap = 'viridis'
        elif channel == 'Shape index map':
            img = self.image.mask_shape_indexed
            cmap = 'viridis'
        elif channel == 'Crude segmentation':
            img = self.image.mask_labeled_clusters
            cmap = 'viridis'
        clear_output()
        display(self.tabview)
        tifffile.imshow(img, cmap=cmap)
        plt.axis('off')

    def record_configuration_handler(self, b):
        currenttime = time.ctime()
        currenttime = '_'.join(currenttime.split(" "))
        self.clusterconfig.set_default_config_data(self.configurator.config)
        print("Current configurations saved!")
        with open('configuration_{}.ini'.format(currenttime), 'w') as configfile:
            self.configurator.config.write(configfile)

    def import_image(self, b):
        if self.fileformat == 0:
            if self.image_filename.endswith('.nd2'):
                self.image = om.Image()
                self.image.read_nd2_file(self.image_filename)
            else:
                print('Invalid file format, please try again!')
        elif self.fileformat == 1:
            if self.image_filename.endswith('.tif'):
                self.image = om.Image()
                self.image.read_tiff_file(self.image_filename)
            else:
                print('Invalid file format, please try again!')
        elif self.fileformat == 2:
            print("Format not supported yet, sorry!")
        else:
            print("No images found!")
        if self.image != None:
            self.channel_dropdown.options = self.image.channels
            print('Successfully imported {}'.format(self.image_filename))

    def channel_dropdown_handler(self, change):
        self.image.mask_channel_name = change['new']
        self.mask_channel_name = change['new']
        clear_output()
        display(self.tabview)
        tifffile.imshow(self.image.data[change['new']], cmap='gist_gray')
        plt.axis('off')

    def setmask_handler(self, b):
        self.channel_dropdown.disabled = True
        print('Channel {} will be used for cell segmentation.'. \
              format(self.image.mask_channel_name))
        print('Please proceed to part II.')


class Preset_configurations_default:
    def __init__(self, configfile=None):
        self.config = configparser.ConfigParser()
        if configfile == None:
            self.config.read("OMEGA/configuration.ini")
            self.configurated = True
        else:
            self.config.read(configfile)
            self.configurated = True