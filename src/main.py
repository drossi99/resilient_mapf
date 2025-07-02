from src.view.controller import MainGUIController
import customtkinter as ctk
from pathlib import  Path
import os

if __name__ == '__main__':
    theme_path = str(Path("themes/midnight.json"))
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme(theme_path)
    ctk.set_window_scaling(1.0)
    ctk.set_widget_scaling(1.0)

    app = MainGUIController()
    app.run()
