import gtk


class Toggle_Box(gtk.HBox):

    HANDLER = "customChangedSignal"

    def __init__(self, homogenous=False, spacing=0, size=None, elements=None):

        super(Toggle_Box, self).__init__(homogenous, spacing)

        self._changedSignal = None
        self._changedActive = True
        self._changedArgs = None

        if elements is not None:
            self.set_elements(elements)

        if size is not None:
            self.set_size(size)

    def connect(self, signal, func, *args):

        if signal == "changed":

            self._changedSignal = func
            self._changedArgs = args
            return self.HANDLER

        else:

            return super(Toggle_Box, self).connect(signal, func, *args)

    def handler_block(self, handler):

        if handler is self.HANDLER:
            self._changedActive = False

        else:

            super(Toggle_Box, self).handler_block(handler)

    def handler_unblock(self, handler):

        if handler is self.HANDLER:
            self._changedActive = True
        else:
            super(Toggle_Box, self).handler_unblock(handler)

    def emit(self, handler):

        if handler is self.HANDLER:
            if self._changedActive and self._changedSignal is not None:
                self._changedSignal(self, *self._changedArgs)
        else:
            super(Toggle_Box, self).emit(handler)

    def set_elements(self, elements):

        for c in self.get_children():
            c.destroy()

        for e in elements:
            tb = gtk.ToggleButton(label=str(e))
            tb.connect("toggled", self._set_toggles)
            self.pack_start(tb, expand=False, fill=False)
            tb.show()

        self.emit(self.HANDLER)

    def set_size(self, size):

        try:
            size = int(size)
        except ValueError:
            print self, " got bad size, must be int"
            return

        self.set_elements(range(1, size + 1))

    def _set_toggles(self, widget):

        if (widget.get_active()):

            for i, e in enumerate(self.get_children()):
                if e != widget:
                    e.set_active(False)
            self.emit(self.HANDLER)

    def get_active(self):

        for i, e in enumerate(self.get_children()):
            if e.get_active():
                return i

    def set_active(self, index):

        if index in range(1, len(self.children) + 1):
            self.get_children()[index - 1].set_active(True)
