import tkinter as tk
import customtkinter as ctk

class CustomSpinbox(ctk.CTkFrame):
    def __init__(self, master, width=100, height=32, step_size=1, from_=0, to=100,
                 fg_color="transparent", **kwargs):
        super().__init__(master, width=width, height=height, fg_color=fg_color, **kwargs)

        self.step_size = step_size
        self.from_ = from_
        self.to = to

        self.grid_columnconfigure(0, weight=1)

        # Entry
        self.entry = ctk.CTkEntry(
            self,
            width=width - (height * 2),
            height=height,
            border_width=0,
            justify='left'
        )
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        self.entry.insert(0, str(from_))

        # Buttons Frame
        self.button_frame = ctk.CTkFrame(self, width=height * 2, height=height, fg_color="transparent")
        self.button_frame.grid(row=0, column=1, sticky="e")
        self.button_frame.grid_columnconfigure((0, 1), weight=1)
        self.button_frame.grid_rowconfigure((0, 1), weight=1)

        # Up Button
        self.up_btn = ctk.CTkButton(
            self.button_frame,
            width=height,
            height=height // 2,
            text="▲",
            command=lambda: self._update_value(self.step_size),
            anchor="center",
            font=("Arial", int(height // 4))
        )
        self.up_btn.grid(row=0, column=0, sticky="nsew")#, padx=(1, 0))


        # Down Button
        self.down_btn = ctk.CTkButton(
            self.button_frame,
            width=height,
            height=height // 2,
            text="▼",
            command=lambda: self._update_value(-self.step_size),
            anchor="center",
            font=("Arial", int(height // 4))
        )
        self.down_btn.grid(row=1, column=0, sticky="nsew")#, padx=(1, 0), pady=(1, 0))

        # Validation
        vcmd = (self.register(self._validate_input), '%P')
        self.entry.configure(validate='key', validatecommand=vcmd)

        # Mouse wheel binding
        self.entry.bind("<MouseWheel>", self._on_mouse_wheel)
        self.button_frame.bind("<MouseWheel>", self._on_mouse_wheel)

    def _validate_input(self, new_value):
        if new_value.strip() == "":
            return True
        try:
            value = float(new_value)
            if self.from_ <= value <= self.to:
                return True
            return False
        except:
            return False

    def _update_value(self, amount):
        current_value = float(self.entry.get())
        new_value = current_value + amount
        if new_value >= self.from_ and new_value <= self.to:
            self.entry.delete(0, 'end')
            self.entry.insert(0, str(new_value))

    def _on_mouse_wheel(self, event):
        if event.delta > 0:
            self._update_value(self.step_size)
        else:
            self._update_value(-self.step_size)

    def get(self):
        return float(self.entry.get())

    def set(self, value):
        self.entry.delete(0, 'end')
        self.entry.insert(0, str(value))