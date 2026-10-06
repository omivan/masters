
import chess
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from Code.lc_engine import Lc0Engine

ENGINE_PATH = "./lc0"
PIECE_SIZE = 80
MARGIN = 40
LABEL_MARGIN = 20

BUTTON_STYLE = {
    "bg": "#444444",  # Dark gray background
    "fg": "white",    # White text
    "activebackground": "#555555",  # Hover effect
    "activeforeground": "white",
    "font": ("Noto Sans", 13),
    "relief": FLAT,
    "padx": 20,
    "pady": 10,
    "borderwidth": 0,
    "highlightthickness": 0,
    "disabledforeground": "#777777"
}


def setup_global_styles():
    style = ttk.Style()
    style.theme_use("clam")

    style.configure("Dark.TButton",
        background="#444",
        foreground="white",
        font=("Noto Sans", 13),
        padding=10
    )
    style.map("Dark.TButton",
        background=[("active", "#555")],
        foreground=[("active", "white")]
    )

    style.configure("Red.TButton",
        background="#e74c3c",
        foreground="white",
        font=("Noto Sans", 13),
        padding=10
    )
    style.map("Red.TButton",
        background=[("active", "#c0392b")],
        foreground=[("active", "white")]
    )


class ChessApp:
    def __init__(self, root, player_color, weights_path):
        self.root = root
        self.move_number = 1
        self.board = chess.Board()
        self.selected_square = None
        self.player_color = player_color
        self.flipped = self.player_color == "b"
        self.weights_path = weights_path

        # Extract bot level from weights path
        self.bot_level = "Невідомий"
        if "maia-1600" in weights_path:
            self.bot_level = "Середній (1600-1700)"
        elif "maia-2200" in weights_path:
            self.bot_level = "Важкий (2000+)"

        # Update window title to include bot level
        root.title(f"Шахова партія - Бот: {self.bot_level}")

        # Create a frame for info at the top
        button_frame = Frame(root, bg="#2c3e50")
        button_frame.pack(pady=5, fill=X)

        # # Add bot level info label
        # self.bot_info_label = Label(
        #     button_frame,
        #     text=f"Рівень бота: {self.bot_level}",
        #     bg="#2c3e50",
        #     fg="white",
        #     font=("Helvetica", 12)
        # )
        # self.bot_info_label.pack(pady=5)

        canvas_width = 8 * PIECE_SIZE + 2 * MARGIN
        canvas_height = 8 * PIECE_SIZE + 2 * MARGIN + LABEL_MARGIN
        self.canvas = Canvas(root, width=canvas_width, height=canvas_height)
        self.canvas.pack()

        # Налаштування стилю для кнопок
        setup_global_styles()

        # Створення кнопки "Здатися" над дошкою
        top_button_frame = Frame(root, bg="#2c3e50")
        top_button_frame.pack(pady=(5, 0))

        self.resign_button = ttk.Button(
            top_button_frame,
            text="Здатися",
            command=self.resign_game,
            style="Dark.TButton",
            width=15,
            takefocus=False
        )
        self.resign_button.pack(pady=5)

        self.piece_images = self.load_piece_images()
        self.piece_ids = {}
        self.drag_data = {"piece": None, "from": None, "image_id": None, "start_coords": (0, 0), "legal_targets": []}

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_drop)

        self.engine = Lc0Engine(ENGINE_PATH, weights_path)

        self.draw_board()

        if self.player_color == "b":
            self.engine_move()

    def player_to_move(self) -> bool:
        """Return True if it’s the human’s turn."""
        return (self.board.turn and self.player_color == "w") or \
            (not self.board.turn and self.player_color == "b")
    def load_piece_images(self):
        pieces = {}
        for color in ['w', 'b']:
            for piece in ['K', 'Q', 'R', 'B', 'N', 'P']:
                name = f"{color}{piece}"
                img = Image.open(f"img/{name}.png").resize((PIECE_SIZE, PIECE_SIZE), Image.LANCZOS)
                pieces[name] = ImageTk.PhotoImage(img)
        return pieces

    def draw_board(self):
        light = "#dcdcdc"
        dark = "#666666"
        temp_dragged = self.drag_data.get("from")
        dragged_id = self.drag_data.get("image_id")
        self.canvas.delete("all")
        self.piece_ids.clear()
        if temp_dragged is not None and dragged_id:
            self.piece_ids[temp_dragged] = dragged_id

        for rank in range(8):
            for file in range(8):
                r = 7 - rank if not self.flipped else rank
                f = file if not self.flipped else 7 - file
                x = MARGIN + f * PIECE_SIZE
                y = MARGIN + r * PIECE_SIZE
                color = dark if (rank + file) % 2 == 0 else light
                self.canvas.create_rectangle(x, y, x + PIECE_SIZE, y + PIECE_SIZE, fill=color, outline="")

        for rank in range(8):
            for file in range(8):
                square = chess.square(file, rank)
                if square == temp_dragged:
                    continue
                r = 7 - rank if not self.flipped else rank
                f = file if not self.flipped else 7 - file
                x = MARGIN + f * PIECE_SIZE
                y = MARGIN + r * PIECE_SIZE
                piece = self.board.piece_at(square)
                if piece:
                    name = f"{'w' if piece.color == chess.WHITE else 'b'}{piece.symbol().upper()}"
                    image_id = self.canvas.create_image(x + PIECE_SIZE//2, y + PIECE_SIZE//2, anchor=CENTER, image=self.piece_images[name])
                    self.piece_ids[square] = image_id

        for i in range(8):
            rank_label = str(i + 1) if self.flipped else str(8 - i)
            y = MARGIN + i * PIECE_SIZE + PIECE_SIZE // 2
            self.canvas.create_text(MARGIN // 2, y, text=rank_label, font=("Arial", 14))

        for i in range(8):
            file_label = chr(ord('h') - i) if self.flipped else chr(ord('a') + i)
            x = MARGIN + i * PIECE_SIZE + PIECE_SIZE // 2
            y = 8 * PIECE_SIZE + MARGIN + LABEL_MARGIN // 2
            self.canvas.create_text(x, y, text=file_label, font=("Arial", 14))

        # Розміщення кнопки "Здатися" на дошці в правому нижньому куті (h1 для білих, a8 для чорних)
        if self.player_color == "w":
            # Для білих - правий нижній кут (h1)
            button_x = MARGIN + 7 * PIECE_SIZE + PIECE_SIZE // 2
            button_y = MARGIN + 7 * PIECE_SIZE + PIECE_SIZE // 2
        else:
            # Для чорних - лівий верхній кут (a8)
            button_x = MARGIN + 0 * PIECE_SIZE + PIECE_SIZE // 2
            button_y = MARGIN + 0 * PIECE_SIZE + PIECE_SIZE // 2


    def on_click(self, event):
        if not self.player_to_move():
            return
        if self.flipped:
            file = 7 - ((event.x - MARGIN) // PIECE_SIZE)
            rank = (event.y - MARGIN) // PIECE_SIZE
        else:
            file = (event.x - MARGIN) // PIECE_SIZE
            rank = 7 - ((event.y - MARGIN) // PIECE_SIZE)

        square = chess.square(file, rank)

        if self.selected_square is not None:
            move = chess.Move(self.selected_square, square)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                self.draw_board()

                # Check if the game is over after the player's move
                if self.board.is_game_over():
                    # Додаємо затримку перед показом діалогу завершення гри,
                    # щоб користувач міг побачити останній хід
                    self.root.after(1000, self.end_game)
                else:
                    self.engine_move()
                return
            else:
                self.selected_square = None
                self.draw_board()

        piece = self.board.piece_at(square)
        if piece:
            is_players_piece = (self.player_color == 'w' and piece.color == chess.WHITE) or (self.player_color == 'b' and piece.color == chess.BLACK)

            if is_players_piece:
                self.selected_square = square

                # Check for legal moves from this square
                legal_moves_from_square = [m for m in self.board.legal_moves if m.from_square == square]

                # Get the image ID for the piece
                image_id = self.piece_ids.get(square)

                # Always set up drag data, even if there are no legal moves
                if image_id is not None:
                    # Calculate display coordinates
                    display_file = file if not self.flipped else 7 - file
                    display_rank = 7 - rank if not self.flipped else rank

                    # Set up drag data
                    self.drag_data = {
                        "from": square,
                        "piece": piece,
                        "legal_targets": [m.to_square for m in legal_moves_from_square],
                        "image_id": image_id,
                        "start_coords": (
                            MARGIN + display_file * PIECE_SIZE + PIECE_SIZE // 2,
                            MARGIN + display_rank * PIECE_SIZE + PIECE_SIZE // 2
                        )
                    }
                    self.canvas.tag_raise(image_id)

    def on_drag(self, event):
        if self.drag_data["image_id"]:
            self.canvas.coords(self.drag_data["image_id"], event.x, event.y)

    def on_drop(self, event):
        if not self.player_to_move():
            return

        if not self.drag_data["from"] or not self.drag_data["image_id"]:
            return

        if self.flipped:
            file = 7 - ((event.x - MARGIN) // PIECE_SIZE)
            rank = (event.y - MARGIN) // PIECE_SIZE
        else:
            file = (event.x - MARGIN) // PIECE_SIZE
            rank = 7 - ((event.y - MARGIN) // PIECE_SIZE)

        to_square = None
        if (0 <= file < 8 and 0 <= rank < 8):
            to_square = chess.square(file, rank)

        from_square = self.drag_data["from"]
        image_id = self.drag_data["image_id"]
        start_x, start_y = self.drag_data["start_coords"]

        if to_square is not None and to_square in self.drag_data["legal_targets"]:
            # Calculate the target coordinates for animation
            to_file = chess.square_file(to_square)
            to_rank = chess.square_rank(to_square)
            r = 7 - to_rank if not self.flipped else to_rank
            f = to_file if not self.flipped else 7 - to_file
            target_x = MARGIN + f * PIECE_SIZE + PIECE_SIZE // 2
            target_y = MARGIN + r * PIECE_SIZE + PIECE_SIZE // 2

            # Animate the piece movement
            self.animate_piece_movement(image_id, event.x, event.y, target_x, target_y, 
                                       lambda: self.complete_move(from_square, to_square))
        else:
            # Animate the piece returning to its original position
            self.animate_piece_movement(image_id, event.x, event.y, start_x, start_y, 
                                       lambda: self.reset_drag_data())

    def animate_piece_movement(self, image_id, start_x, start_y, target_x, target_y, callback, steps=10, delay=10):
        """Animate a piece moving from current position to target position"""
        # Calculate step sizes
        dx = (target_x - start_x) / steps
        dy = (target_y - start_y) / steps

        def animate_step(step):
            if step < steps:
                # Calculate new position
                new_x = start_x + dx * step
                new_y = start_y + dy * step
                # Update piece position
                self.canvas.coords(image_id, new_x, new_y)
                # Schedule next step
                self.root.after(delay, lambda: animate_step(step + 1))
            else:
                # Final position
                self.canvas.coords(image_id, target_x, target_y)
                # Call the callback function when animation is complete
                callback()

        # Start animation
        animate_step(1)

    def complete_move(self, from_square, to_square):
        self.selected_square = None
        """Complete the move after animation"""
        move = chess.Move(from_square, to_square)
        self.board.push(move)
        if self.board.turn == chess.WHITE:
            self.move_number += 1
        self.drag_data = {"piece": None, "from": None, "image_id": None, "start_coords": (0, 0), "legal_targets": []}
        self.draw_board()
        # self.update_status()
        if self.board.is_game_over():
            # Додаємо затримку перед показом діалогу завершення гри,
            # щоб користувач міг побачити останній хід
            self.root.after(1000, self.end_game)
        else:
            self.engine_move()

    def reset_drag_data(self):
        """Reset drag data after animation"""
        self.drag_data = {"piece": None, "from": None, "image_id": None, "start_coords": (0, 0), "legal_targets": []}

    def engine_move(self):
        if self.board.is_game_over():
            self.end_game()
            return
        result = self.engine.play(self.board, nodes=1)

        # Додаємо невеликий ділей перед відображенням ходу бота (0.2 секунди)
        def apply_bot_move():
            self.board.push(result.move)
            self.draw_board()
            if self.board.is_game_over():
                # Додаємо затримку перед показом діалогу завершення гри,
                # щоб користувач міг побачити останній хід
                self.root.after(1000, self.end_game)

        # Запускаємо функцію з затримкою в 200 мс
        self.root.after(200, apply_bot_move)

    def resign_game(self):
        winner = "Чорні" if self.player_color == "w" else "Білі"
        messagebox.showinfo("Гра закінчена", f"Ви здалися. {winner} парамагають.")
        self.ask_new_game()

    def end_game(self):
        result = self.board.result()
        result_text = "1-0 (Білі перемагають)" if result == "1-0" else "0-1 (Чорні перемагають)" if result == "0-1" else "1/2-1/2 (Нічия)"
        messagebox.showinfo("Гра закінчена", f"Гра закінчена. Результат: {result_text}")
        self.ask_new_game()

    def ask_new_game(self):
        # Create a custom dialog with three buttons
        dialog = Toplevel(self.root)
        dialog.title("Опції гри")
        dialog.geometry("440x280")
        dialog.configure(bg="#2c3e50")
        dialog.transient(self.root)  # Make dialog modal
        dialog.grab_set()  # Make dialog modal
        dialog.resizable(False, False)  # Prevent resizing

        # Center the dialog on the parent window
        self.root.update_idletasks()
        parent_x = self.root.winfo_rootx()
        parent_y = self.root.winfo_rooty()
        parent_width = self.root.winfo_width()
        parent_height = self.root.winfo_height()

        width = 440
        height = 280
        x = parent_x + (parent_width // 2) - (width // 2)
        y = parent_y + (parent_height // 2) - (height // 2)
        dialog.geometry(f'{width}x{height}+{x}+{y}')

        # Add a label
        Label(dialog, text="Що далі?", bg="#2c3e50", fg="white",
              font=("Helvetica", 14, "bold")).pack(pady=20)

        # Variable to store the result
        result = StringVar(value="")

        # Function to set the result and close the dialog
        def set_result(value):
            result.set(value)
            dialog.destroy()

        # Create a frame for the buttons
        button_frame = Frame(dialog, bg="#2c3e50")
        button_frame.pack(pady=10, padx=30, fill=X)

        # Button style
        button_style = {
            "bg": "#444",
            "fg": "white",
            "font": ("Noto Sans", 12),
            "relief": FLAT,
            "padx": 15,
            "pady": 8,
            "borderwidth": 0,
            "highlightthickness": 0,
            "activebackground": "#555",
            "activeforeground": "white",
            "width": 25  # Fixed width for all buttons
        }

        # Add the buttons
        setup_global_styles()

        # Replace buttons with this
        ttk.Button(button_frame, text="Грати знову (ті самі налаштування)",
                   command=lambda: set_result("same"),
                   style="Dark.TButton").pack(fill=X, pady=5)

        ttk.Button(button_frame, text="Повернутися до головного меню",
                   command=lambda: set_result("menu"),
                   style="Dark.TButton").pack(fill=X, pady=5)
        # Wait for the dialog to be closed
        self.root.wait_window(dialog)

        # Handle the result
        if result.get() == "same":  # Play again with same settings
            # Store current settings
            current_color = self.player_color
            current_weights = self.weights_path

            # Quit the engine and destroy the window
            self.engine.quit()
            self.root.destroy()

            # Create a new game with the same settings
            root = Tk()
            root.configure(bg="#2c3e50")
            root.resizable(False, False)  # Prevent window resizing
            app = ChessApp(root, player_color=current_color, weights_path=current_weights)
            root.protocol("WM_DELETE_WINDOW", app.on_close)
            root.mainloop()

        else:  # Go to main menu (default option)
            self.engine.quit()
            self.root.destroy()
            config_window = create_config_window()
            config_window.mainloop()

    def on_close(self):
        self.engine.quit()
        self.root.destroy()

def create_config_window():
    config_window = Tk()
    config_window.title("Головне меню шахового бота")
    config_window.geometry("700x700")
    config_window.resizable(False, False)  # Prevent window resizing
    config_window.configure(bg="#2c3e50")

    # Setup global styles for buttons
    setup_global_styles()

    Label(config_window, text="Ласкаво просимо до шахового бота", bg="#2c3e50", fg="white", font=("Helvetica", 20, "bold")).pack(pady=20)

    # Add a spacer frame between welcome message and color selection
    spacer_between_welcome_color = Frame(config_window, bg="#2c3e50", height=30)
    spacer_between_welcome_color.pack(fill=X)

    color_frame = Frame(config_window, bg="#2c3e50")
    Label(color_frame, text="Виберіть колір яким бажаєте грати:", bg="#2c3e50", fg="white", font=("Helvetica", 14)).pack(anchor=W)
    color_var = StringVar(value="w")
    Radiobutton(color_frame, text="Білі", variable=color_var, value="w", bg="#2c3e50", fg="white", selectcolor="#34495e", font=("Helvetica", 12)).pack(anchor=W, pady=5)
    Radiobutton(color_frame, text="Чорні", variable=color_var, value="b", bg="#2c3e50", fg="white", selectcolor="#34495e", font=("Helvetica", 12)).pack(anchor=W)
    color_frame.pack(pady=20, fill=X, padx=60)

    bot_frame = Frame(config_window, bg="#2c3e50")
    Label(bot_frame, text="Виберіть рівень бота:", bg="#2c3e50", fg="white", font=("Helvetica", 14)).pack(anchor=W)
    weights_var = StringVar(value="./model_files/maia-2200.pb.gz")
    Radiobutton(bot_frame, text="Середній (1600-1700)", variable=weights_var, value="./model_files/maia-1600.pb.gz", bg="#2c3e50", fg="white", selectcolor="#34495e", font=("Helvetica", 12)).pack(anchor=W, pady=5)
    Radiobutton(bot_frame, text="Важкий (2000+)", variable=weights_var, value="./model_files/maia-2200.pb.gz", bg="#2c3e50", fg="white", selectcolor="#34495e", font=("Helvetica", 12)).pack(anchor=W)
    bot_frame.pack(pady=20, fill=X, padx=60)

    # Add a spacer frame to push buttons toward the bottom
    spacer_frame = Frame(config_window, bg="#2c3e50", height=100)
    spacer_frame.pack(fill=X)

    def start_game():
        # Get the selected options
        color = color_var.get()
        weights = weights_var.get()
        if color not in ["w", "b"] or not weights:
            return

        print("Starting game with color:", color, "and weights:", weights)

        config_window.destroy()
        root = Tk()
        root.resizable(False, False)  # Prevent window resizing
        app = ChessApp(root, player_color=color, weights_path=weights)
        root.protocol("WM_DELETE_WINDOW", app.on_close)
        root.mainloop()

    # Create the start button
    start_button = ttk.Button(config_window, text="Почати гру", command=start_game, style="Dark.TButton")
    start_button.pack(pady=15)

    # Create the exit button
    def exit_game():
        config_window.destroy()  # Properly close the window

    exit_button = ttk.Button(config_window, text="Вийти", command=exit_game, style="Red.TButton")
    exit_button.pack(pady=15)

    return config_window

if __name__ == "__main__":
    config_window = create_config_window()
    config_window.mainloop()
