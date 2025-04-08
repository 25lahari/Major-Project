# import tkinter as tk
# from tkinter import messagebox
# from PIL import Image, ImageTk
# import subprocess

# # Function to execute a file
# def run_file(file_path):
#     try:
#         subprocess.run(['python', file_path], check=True)
#         messagebox.showinfo("Success", f"Executed {file_path} successfully!")
#         show_comparison_table()  # Show comparison table after successful execution
#     except subprocess.CalledProcessError as e:
#         messagebox.showerror("Error", f"An error occurred while running {file_path}: {e}")

# # Main window setup
# def main_window():
#     root = tk.Tk()
#     root.title("AI Driven Ransomware Detection using DeepLearning")
#     root.geometry("600x600")
#     root.configure(bg="#ADD8E6")

#     # Title Label
#     title_label = tk.Label(root, text="AI Driven Ransomware Detection using DeepLearning", font=("Helvetica", 36), bg="#ADD8E6", fg="black")
#     title_label.pack(pady=70)

#     # Next Button
#     next_button = tk.Button(root, text="Next", font=("Helvetica", 22), bg="#008000", fg="white", command=lambda: show_second_window(root))
#     next_button.pack()

#     root.mainloop()

# # Second window setup
# def show_second_window(root):
#     root.destroy()

#     second_window = tk.Tk()
#     second_window.title("Choose a Model")
#     second_window.geometry("600x600")
#     second_window.configure(bg="#E3E4FA")

#     # Instruction Label
#     instruction_label = tk.Label(second_window, text="Select a model to execute", font=("Helvetica", 24), bg="#E3E4FA", fg="black")
#     instruction_label.pack(pady=40)

#     # Buttons for each file
#     button_rf = tk.Button(second_window, text="Run RandomForest.py", font=("Helvetica", 22), bg="#2E8B57", fg="white",
#                            command=lambda: run_file("RandomForest.py"))
#     button_rf.pack(pady=20)

#     button_seq = tk.Button(second_window, text="Run Sequential.py", font=("Helvetica", 22), bg="#CC7A8B", fg="white",
#                             command=lambda: run_file("Sequential.py"))
#     button_seq.pack(pady=20)

#     button_xgb = tk.Button(second_window, text="Run XGBoostModel.py", font=("Helvetica", 22), bg="#7E587E", fg="white",
#                             command=lambda: run_file("XGBoostModel.py"))
#     button_xgb.pack(pady=20)

#     second_window.mainloop()

# # Show image display window
# def show_images_window():
#     images_window = tk.Toplevel()
#     images_window.title("Images Display")
#     images_window.geometry("600x600")
#     images_window.configure(bg="#ADD8E6")

#     # Load and display an image (example)
#     try:
#         img = Image.open(r"C:\Users\LAHARI\OneDrive\Desktop\MajorProject\image.jpg")  # Replace with the actual image path
#         img = img.resize((900, 800), Image.Resampling.LANCZOS)
#         img_tk = ImageTk.PhotoImage(img)

#         label = tk.Label(images_window, image=img_tk, bg="#ADD8E6")
#         label.image = img_tk  # Keep a reference to avoid garbage collection
#         label.pack(pady=40)
#     except FileNotFoundError:
#         messagebox.showerror("Error", "Image not found!")

#     images_window.mainloop()

# # Show comparison table window
# def show_comparison_table():
#     table_window = tk.Toplevel()  # Use Toplevel instead of Tk for a new window
#     table_window.title("Comparison Table")
#     table_window.geometry("500x500")
#     table_window.configure(bg="#DCD0FF")

#     # Load and display the comparison table image
#     try:

#         img = Image.open(r"C:\Users\LAHARI\OneDrive\Desktop\MajorProject\compare.jpg")  # Replace with the actual image path
#         img = img.resize((700, 400), Image.Resampling.LANCZOS)
#         img_tk = ImageTk.PhotoImage(img)

#         label = tk.Label(table_window, image=img_tk, bg="#FDD7E4")
#         label.image = img_tk  # Keep a reference to avoid garbage collection
#         label.pack(pady=30)

#         # Next Button
#         next_button = tk.Button(table_window, text="Next", font=("Helvetica", 18), bg="#008000", fg="white", command=show_images_window)
#         next_button.pack(pady=20)
#     except FileNotFoundError:
#         messagebox.showerror("Error", "Comparison table image not found!")

#     table_window.mainloop()

# # Run the main window
# if __name__ == "__main__":
#     main_window()

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import subprocess

# Function to execute a file
def run_file(file_path):
    try:
        subprocess.run(['python', file_path], check=True)
        messagebox.showinfo("Success", f"Executed {file_path} successfully!")
        show_comparison_table()  # Show comparison table after successful execution
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred while running {file_path}: {e}")

# Main window setup
def main_window():
    root = tk.Tk()
    root.title("AI Driven Ransomware Detection using DeepLearning")
    root.state('zoomed')  # Open in fullscreen
    root.configure(bg="#ADD8E6")

    # Title Label
    title_label = tk.Label(root, text="AI Driven Ransomware Detection using DeepLearning", font=("Helvetica", 36), bg="#ADD8E6", fg="black")
    title_label.pack(pady=70)

    # Next Button
    next_button = tk.Button(root, text="Next", font=("Helvetica", 22), bg="#008000", fg="white", command=lambda: show_second_window(root))
    next_button.pack()

    root.mainloop()

# Second window setup
def show_second_window(root):
    root.destroy()

    second_window = tk.Tk()
    second_window.title("Choose a Model")
    second_window.state('zoomed')  # Open in fullscreen
    second_window.configure(bg="#E3E4FA")

    # Instruction Label
    instruction_label = tk.Label(second_window, text="Select a model to execute", font=("Helvetica", 24), bg="#E3E4FA", fg="black")
    instruction_label.pack(pady=40)

    # Buttons for each file
    button_rf = tk.Button(second_window, text="Run RandomForest.py", font=("Helvetica", 22), bg="#2E8B57", fg="white",
                           command=lambda: run_file("RandomForest.py"))
    button_rf.pack(pady=20)

    button_seq = tk.Button(second_window, text="Run Sequential.py", font=("Helvetica", 22), bg="#CC7A8B", fg="white",
                            command=lambda: run_file("Sequential.py"))
    button_seq.pack(pady=20)

    button_xgb = tk.Button(second_window, text="Run XGBoostModel.py", font=("Helvetica", 22), bg="#7E587E", fg="white",
                            command=lambda: run_file("XGBoostModel.py"))
    button_xgb.pack(pady=20)

    second_window.mainloop()

# Show image display window
def show_images_window():
    images_window = tk.Toplevel()
    images_window.title("Images Display")
    images_window.state('zoomed')  # Open in fullscreen
    images_window.configure(bg="#ADD8E6")

    # Load and display an image (example)
    try:
        img = Image.open(r"C:\Users\LAHARI\OneDrive\Desktop\MajorProject\image.jpg")  # Replace with the actual image path
        img = img.resize((900, 800), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)

        label = tk.Label(images_window, image=img_tk, bg="#ADD8E6")
        label.image = img_tk  # Keep a reference to avoid garbage collection
        label.pack(pady=40)
    except FileNotFoundError:
        messagebox.showerror("Error", "Image not found!")

    images_window.mainloop()

# Show comparison table window
def show_comparison_table():
    table_window = tk.Toplevel()  # Use Toplevel instead of Tk for a new window
    table_window.title("Comparison Table")
    table_window.state('zoomed')  # Open in fullscreen
    table_window.configure(bg="#DCD0FF")

    # Load and display the comparison table image
    try:
        img = Image.open(r"C:\Users\LAHARI\OneDrive\Desktop\MajorProject\compare.jpg")  # Replace with the actual image path
        img = img.resize((700, 400), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)

        label = tk.Label(table_window, image=img_tk, bg="#FDD7E4")
        label.image = img_tk  # Keep a reference to avoid garbage collection
        label.pack(pady=30)

        # Next Button
        next_button = tk.Button(table_window, text="Next", font=("Helvetica", 18), bg="#008000", fg="white", command=show_images_window)
        next_button.pack(pady=20)
    except FileNotFoundError:
        messagebox.showerror("Error", "Comparison table image not found!")

    table_window.mainloop()

# Run the main window
if __name__ == "__main__":
    main_window()

