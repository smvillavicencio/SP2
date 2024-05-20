import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog

def showPGN(pgn):
    def copyText():
        text = textBox.get("1.0", "end-1c")
        root.clipboard_clear()
        root.clipboard_append(text)
        messagebox.showinfo("Copy", "Text copied to clipboard.")

    def saveAsPGN():
        text = textBox.get("1.0", "end-1c")
        filePath = filedialog.asksaveasfilename(defaultextension=".pgn", filetypes=[("PGN files", "*.pgn")])
        if filePath:
            with open(filePath, "w") as file:
                file.write(text)
            messagebox.showinfo("Save", f"Text saved as {filePath}")

    # Create the main window
    root = tk.Tk()
    root.title("PGN Viewer")

    # Make the window non-resizable
    root.resizable(False, False)

    # Create a text box with a scrollbar
    textFrame = ttk.Frame(root)
    textFrame.pack(fill=tk.BOTH, expand=True)

    textBox = tk.Text(textFrame, wrap="word", state="disabled")
    textBox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(textFrame, orient=tk.VERTICAL, command=textBox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    textBox.config(yscrollcommand=scrollbar.set)

    textBox.configure(state="normal")
    textBox.insert(tk.END, pgn)
    textBox.configure(state="disabled")

    # Create buttons for copying text and saving as .pgn file
    buttonFrame = ttk.Frame(root)
    buttonFrame.pack(pady=10)

    copyButton = ttk.Button(buttonFrame, text="Copy Text", command=copyText)
    copyButton.pack(side=tk.LEFT, padx=5)

    saveButton = ttk.Button(buttonFrame, text="Save as .pgn", command=saveAsPGN)
    saveButton.pack(side=tk.LEFT, padx=5)

    # Run the main loop
    root.mainloop()