import PySimpleGUI as sg
import cv2
import torch
from utils import *

class App:
    
    def __init__(self, config):
        self.config = config
        self.window1 = self.make_main_window()
        self.window2 = None
        self.webcam_frames_grabber  = None
        self.fps = FPS()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.models = None # self.__set_models()

    def settings_window(self):
        """
            Checkbox Lair
            in another window
        """
        pass
    
    def __set_models(self) -> Models:
        """
            Return dict with jit models
            
        """
        return Models(
            *[ModelWrap(
                 ModelPathes(
                     *self.config['pathes'][model_purpose]), 
                     self.device
                            )
            for model_purpose in ('face', 'hand', 'voice')])

    def make_main_window(self):
        layout = [
        [sg.Image(key = '-IMAGE-')],
        [sg.Text('Im status string, im describe something lol', 
                 key = '-TEXT-', expand_x = True, justification = 'c')],
        [sg.Input('', key='-IN-'), 
         sg.FileBrowse('Local file', key = '-FS-'),
         sg.Button('Process data', key = '-CAMSTREAM-')],
        ]

        return sg.Window('EmotinRecognitionDemo', 
                         layout, location=(800,600), 
                         finalize=True)

    def make_win2():
        layout = [[sg.Text('The second window')],
               [sg.Button('Exit')]]
        return sg.Window('Second Window', layout, finalize=True)

    def run(self):
        try:
            while True: # Event Loop
                window, event, values = sg.read_all_windows(timeout = 1)
                
                if self.webcam_frames_grabber:
                    frame = self.webcam_frames_grabber.read()
                    
                    # some frame processing
                    # processed_frame = self.models['hand_model'](frame) -> np.array
                    processed_frame = frame 
                    
                    imgbytes = cv2.imencode('.png',processed_frame)[1].tobytes()
                    self.window1['-IMAGE-'].update(data = imgbytes)
                    self.window1['-TEXT-'].update(value=int(self.fps.fps()))               
                        
                elif event == '-SOMEEVENT-' and not self.window2:
                    self.window2 = make_win2()

                elif event == '-IN-':          
                    pass
        
                elif (event == '-CAMSTREAM-') and (values['-IN-'] == '') and \
                     (self.webcam_frames_grabber is None):
                    self.webcam_frames_grabber = WebcamVideoStream(src=0).start()
                        
                elif (event == '-CAMSTREAM-') and (self.webcam_frames_grabber != None):
                    self.webcam_frames_grabber.stop()
                    self.webcam_frames_grabber = None
                    self.window1['-IMAGE-'].update(data=None)
                
                elif (event == 'CAMSTREAM-') and (values['-IN-'] != ''):
                    #values['-IN-'] - link valudation
                    #if local file:
                    #if online stream:
                    pass
                
                elif (event == sg.WIN_CLOSED or event == 'Exit'):# or event == 'Exit
                    window.close()
                    
                    if window == self.window1:     # if closing win 1, exit program
                        if self.webcam_frames_grabber != None:
                            self.webcam_frames_grabber.stop()
                            self.webcam_frames_grabber = None
                            self.window1['-IMAGE-'].update(data = None)
                        break
                            
                    elif window == self.window2:   # if closing win 2, mark as close   
                        self.window2 = None                
                
            # window.close()
        except Exception as e:
            if self.webcam_frames_grabber != None:
                self.webcam_frames_grabber.stop()
            sg.popup_error_with_traceback(f'Somethig went wrong:', e)
            
if __name__ == '__main__':
    gui_app = App(parse_cfg('cfg.yaml'))
    gui_app.run()
