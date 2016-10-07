'''
Created on Oct 4, 2016

@author: daniel
'''

import Tkinter as tk
import tkMessageBox

root = tk.Tk()
root.withdraw()
print tkMessageBox.askokcancel("Warning!", "The face recognition model is not trained. Saving may mix up undesired photos with prefiltered photos.")
print tkMessageBox.askyesno("Save Photos",  "Is this person {}?" )

