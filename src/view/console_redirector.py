import sys
import threading

class ThreadSafeConsoleRedirector:
    def __init__(self, textbox, root):
        self.textbox = textbox
        self.root = root
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.buffer = []
        self.lock = threading.Lock()  # Add thread safety
        self._active = True  # Flag to check if redirector is active

    def write(self, message):
        if not self._active:
            self.original_stdout.write(message)
            return

        if not message or message.strip() == "":
            return

        with self.lock:
            if self.textbox is None:
                self.buffer.append(message)
                return

            # Check if textbox still exists
            try:
                if not self.textbox.winfo_exists():
                    self.buffer.append(message)
                    return
            except Exception:
                # If we can't check, assume it doesn't exist
                self.buffer.append(message)
                return

            try:
                self.root.after(0, self._update_console, message)
            except Exception as e:
                # Fallback to original stdout
                self.original_stdout.write(f"[Console redirect failed: {e}] {message}")

    def _update_console(self, message):
        try:
            if self.textbox and self.textbox.winfo_exists():
                self.textbox.insert("end", message)
                self.textbox.see("end")
                self._flush_buffer()
        except Exception as e:
            self.original_stdout.write(f"[GUI update failed: {e}] {message}")

    def _flush_buffer(self):
        if self.buffer and self.textbox and self.textbox.winfo_exists():
            try:
                for msg in self.buffer:
                    self.textbox.insert("end", msg)
                self.buffer.clear()
                self.textbox.see("end")
            except Exception:
                pass

    def flush(self):
        try:
            if hasattr(self.original_stdout, 'flush'):
                self.original_stdout.flush()
        except Exception as e:
            print(f"[Warning] Console flush failed: {e}", file=sys.__stderr__)

    def close(self):
        self._active = False
        self.restore()

    def restore(self):
        try:
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
        except Exception as e:
            print(f"[Warning] Failed to restore stdout: {e}", file=sys.__stderr__)