"""
ui/spinbox.py
Custom numeric spinbox widget built on top of customtkinter.
Supports mouse wheel, keyboard input validation, and min/max clamping.
"""

import customtkinter as ctk


class CustomSpinbox(ctk.CTkFrame):
    """
    A numeric spinbox widget with increment/decrement buttons and
    direct keyboard entry with validation.

    Parameters
    ----------
    master:     Parent widget.
    width:      Total widget width in pixels.
    height:     Widget height in pixels.
    step_size:  Amount to increment/decrement per click or scroll.
    from_:      Minimum allowed value (inclusive).
    to:         Maximum allowed value (inclusive).
    fg_color:   Frame background ("transparent" by default).
    """

    def __init__(
        self,
        master,
        width: int = 100,
        height: int = 32,
        step_size: int | float = 1,
        from_: int | float = 0,
        to: int | float = 100,
        fg_color: str = "transparent",
        **kwargs,
    ):
        super().__init__(master, width=width, height=height, fg_color=fg_color, **kwargs)

        self.step_size = step_size
        self.from_ = from_
        self.to = to

        self.grid_columnconfigure(0, weight=1)

        # --- Entry ---
        self.entry = ctk.CTkEntry(
            self,
            width=width - (height * 2),
            height=height,
            border_width=0,
            justify="center",
        )
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        self.entry.insert(0, str(from_))

        # --- Button frame ---
        self.button_frame = ctk.CTkFrame(
            self, width=height * 2, height=height, fg_color="transparent"
        )
        self.button_frame.grid(row=0, column=1, sticky="e")
        self.button_frame.grid_columnconfigure((0, 1), weight=1)
        self.button_frame.grid_rowconfigure((0, 1), weight=1)

        btn_font = ("Arial", max(8, int(height // 4)))

        self.up_btn = ctk.CTkButton(
            self.button_frame,
            width=height,
            height=height // 2,
            text="▲",
            command=lambda: self._update_value(self.step_size),
            anchor="center",
            font=btn_font,
        )
        self.up_btn.grid(row=0, column=0, sticky="nsew", padx=(1, 0))

        self.down_btn = ctk.CTkButton(
            self.button_frame,
            width=height,
            height=height // 2,
            text="▼",
            command=lambda: self._update_value(-self.step_size),
            anchor="center",
            font=btn_font,
        )
        self.down_btn.grid(row=1, column=0, sticky="nsew", padx=(1, 0), pady=(1, 0))

        # --- Validation ---
        vcmd = (self.register(self._validate_input), "%P")
        self.entry.configure(validate="key", validatecommand=vcmd)

        # --- Mouse wheel ---
        self.entry.bind("<MouseWheel>", self._on_mouse_wheel)
        self.button_frame.bind("<MouseWheel>", self._on_mouse_wheel)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_input(self, new_value: str) -> bool:
        """Allow empty string (mid-edit) or any number within [from_, to]."""
        if new_value.strip() == "":
            return True
        try:
            value = float(new_value)
            return self.from_ <= value <= self.to
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Value manipulation
    # ------------------------------------------------------------------

    def _update_value(self, amount: int | float) -> None:
        try:
            current = float(self.entry.get())
        except ValueError:
            current = self.from_
        new_value = current + amount
        if self.from_ <= new_value <= self.to:
            self.entry.delete(0, "end")
            self.entry.insert(0, str(new_value))

    def _on_mouse_wheel(self, event) -> None:
        self._update_value(self.step_size if event.delta > 0 else -self.step_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_state(self, state: str) -> None:
        """Enable or disable all interactive child widgets."""
        self.entry.configure(state=state)
        self.up_btn.configure(state=state)
        self.down_btn.configure(state=state)

    def get(self) -> float:
        """Return the current value as a float."""
        try:
            return float(self.entry.get())
        except ValueError:
            return float(self.from_)

    def set(self, value: int | float) -> None:
        """Set the spinbox to *value* (clamped to [from_, to])."""
        clamped = max(self.from_, min(self.to, value))
        self.entry.delete(0, "end")
        self.entry.insert(0, str(clamped))
