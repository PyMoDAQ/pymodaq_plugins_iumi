from pymodaq.utils.daq_utils import ThreadCommand, getLineInfo
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.parameter import Parameter
from pymodaq_plugins_orsay.daq_viewer_plugins.plugins_2D.daq_2Dviewer_OrsayCamera import DAQ_2DViewer_OrsayCamera
from pymodaq_plugins_iumi.hardware.pinemanalysis import PinemAnalysis
import numpy as np

from pathlib import Path
import os

file_path = os.path.dirname(os.path.abspath(__file__))

cnn_folder = Path(file_path) / 'cnns'
cnn_files = [str(file) for file in cnn_folder.glob('*.h5')]


class DAQ_2DViewer_PinemAnalysis(DAQ_2DViewer_OrsayCamera):
    """ Instrument plugin class for a 2D viewer.
    
    This object inherits all functionalities to communicate with PyMoDAQ’s DAQ_Viewer module through inheritance via
    DAQ_Viewer_base. It makes a bridge between the DAQ_Viewer module and the Python wrapper of a particular instrument.

    TODO Complete the docstring of your plugin with:
        * The set of instruments that should be compatible with this instrument plugin.
        * With which instrument it has actually been tested.
        * The version of PyMoDAQ during the test.
        * The version of the operating system.
        * Installation instructions: what manufacturer’s drivers should be installed to make it run?

    Attributes:
    -----------
    controller: object
        The particular object that allow the communication with the hardware, in general a python wrapper around the
         hardware library.
         
    # TODO add your particular attributes here if any

    """

    def ini_detector(self, controller=None):
        """Detector communication initialization

        Parameters
        ----------
        controller: (object)
            custom object of a PyMoDAQ plugin (Slave case). None if only one actuator/detector by controller
            (Master case)

        Returns
        -------
        info: str
        initialized: bool
            False if initialization failed otherwise True
        """
        info, initialized = super().ini_detector(controller)
        self.pinem_model = PinemAnalysis(cnn_files[0])
        return info, initialized

    def emit_data(self):
        """ Method used to emit data obtained by dataUnlocker callback.
        """
        try:
            self.ind_grabbed += 1
            # print(self.ind_grabbed)
            if self.settings['camera_mode_settings', 'camera_mode'] == "Camera":
                if self.camera_done:
                    if self.settings['image_size', 'Nx'] == 1:
                        self.y_axis.index = 0
                        axis = [self.y_axis]
                    elif self.settings['image_size', 'Ny'] == 1:
                        self.x_axis.index = 0
                        axis = [self.x_axis]
                    else:
                        self.y_axis.index = 0
                        self.x_axis.index = 1
                        axis = [self.x_axis, self.y_axis]
                    summed_data = np.atleast_1d(np.squeeze(self.data.reshape(
                                             (self.settings['image_size', 'Ny'],
                                              self.settings['image_size', 'Nx'])))).sum(axis = 0)

                    g = self.pinem_model.predict(summed_data, False)
                    self.dte_signal.emit(
                        DataToExport('OrsayCamera', data=
                        [DataFromPlugins(name='g value',
                                         data=[np.array([g[0][0]])],
                                         dim='Data0D', labels=['g pred']),
                         DataFromPlugins(name=f"Camera {self.settings['model']}",
                                         data=[np.atleast_1d(np.squeeze(self.data.reshape(
                                             (self.settings['image_size', 'Ny'],
                                              self.settings['image_size', 'Nx']))))],
                                         axes=axis)]))

            else:  # spim mode
                # print("spimmode")
                if self.spectrum_done:
                    # print("spectrum done")
                    data = DataToExport('OrsaySPIM', data=
                    [DataFromPlugins(name='SPIM ',
                                     data=[np.atleast_1d(np.squeeze(self.spimdata.reshape(
                                         (self.settings['image_size', 'Nx'],
                                          self.settings['camera_mode_settings', 'spim_y'],
                                          self.settings['camera_mode_settings', 'spim_x']))))],
                                     dim='DataND'),
                     DataFromPlugins(name='Spectrum',
                                     data=[self.spectrumdata],
                                     dim='Data1D')
                     ])
                    if not self.spim_done:
                        self.spectrum_done = False
                        self.dte_signal_temp.emit(data)
                    elif self.spim_done:
                        # print('spimdone')
                        self.dte_signal.emit(data)
                    self.spectrum_done = False
                    self.spim_done = False

        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [getLineInfo() + str(e), 'log']))


if __name__ == '__main__':
    main(__file__, init=False)
