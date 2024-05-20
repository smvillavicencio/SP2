import argparse
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

from integrated_detection import detect
from save import showPGN
from utilities import addNewLine 

def submit():
    eventName = eventNameEntry.get()
    date = dateEntry.get()
    whitePlayerName = whitePlayerNameEntry.get()
    blackPlayerName = blackPlayerNameEntry.get()
    whitePlayerElo = whitePlayerEloEntry.get()
    blackPlayerElo = blackPlayerEloEntry.get()
    whiteSide = whiteSideVar.get()

    if not whitePlayerName or not blackPlayerName:
        messagebox.showerror("Input Error", "Both player names are required.")
        return
    
    if not whiteSide:
        messagebox.showerror("Input Error", "White side is required.")
        return

    header = ""

    if eventName:
        header += f"[Event \"{eventName}\"]\n"
    if date:
        header += f"[Date \"{date}\"]\n"
    if whitePlayerName:
        header += f"[White \"{whitePlayerName}\"]\n"
    if blackPlayerName:
        header += f"[Black \"{blackPlayerName}\"]\n"
    if whitePlayerElo:
        header += f"[WhiteElo \"{whitePlayerElo}\"]\n"
    if blackPlayerElo:
        header += f"[BlackElo \"{blackPlayerElo}\"]\n"

    print(header, whiteSide)
    root.destroy()

    result, san = detect(opt.source, whiteSide)
    
    pgn = f"{header}\n{addNewLine(san + result)}"
    print(f"FEN:\n{header}\n{pgn}")
    showPGN(pgn)


def foo():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=0, help='webcam number')
    opt = parser.parse_args()


    # Create the main window
    root = tk.Tk()
    root.title("Chess Game Details")

    # Make the window non-resizable
    root.resizable(False, False)

    # Disable the close button
    root.protocol("WM_DELETE_WINDOW", foo)

    # Create and place the labels and entries
    labels = [
        "Event Name:",
        "Date (YYYY.MM.DD):",
        "White Player Name:*",
        "Black Player Name:*",
        "White Player ELO:",
        "Black Player ELO:",
        "White Side (Camera POV):*"
    ]

    for idx, text in enumerate(labels):
        label = ttk.Label(root, text=text)
        label.grid(row=idx, column=0, padx=10, pady=5, sticky=tk.W)

    eventNameEntry = ttk.Entry(root)
    eventNameEntry.grid(row=0, column=1, padx=10, pady=5)

    dateEntry = ttk.Entry(root)
    dateEntry.grid(row=1, column=1, padx=10, pady=5)

    whitePlayerNameEntry = ttk.Entry(root)
    whitePlayerNameEntry.grid(row=2, column=1, padx=10, pady=5)

    blackPlayerNameEntry = ttk.Entry(root)
    blackPlayerNameEntry.grid(row=3, column=1, padx=10, pady=5)

    whitePlayerEloEntry = ttk.Entry(root)
    whitePlayerEloEntry.grid(row=4, column=1, padx=10, pady=5)

    blackPlayerEloEntry = ttk.Entry(root)
    blackPlayerEloEntry.grid(row=5, column=1, padx=10, pady=5)

    whiteSideVar = tk.StringVar(value="0")
    whiteSideLeft = ttk.Radiobutton(root, text="Left", variable=whiteSideVar, value=0)
    whiteSideLeft.grid(row=6, column=1, padx=10, pady=5, sticky=tk.W)
    whiteSideRight = ttk.Radiobutton(root, text="Right", variable=whiteSideVar, value=1)
    whiteSideRight.grid(row=6, column=1, padx=10, pady=5, sticky=tk.E)

    submitButton = ttk.Button(root, text="Submit", command=submit)
    submitButton.grid(row=7, columnspan=2, pady=10)

    # Run the main loop
    root.mainloop()



    