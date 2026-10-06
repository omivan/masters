import subprocess
import time
import select
import re
import chess
import threading

class Lc0Engine:
    def __init__(self, engine_path, weights_path):
        """Initialize the lc0 engine process."""
        self.process = subprocess.Popen(
            [engine_path, "--weights=" + weights_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Start a thread to read and print stderr
        self.stderr_thread = threading.Thread(target=self._read_stderr)
        self.stderr_thread.daemon = True
        self.stderr_thread.start()

    def _read_stderr(self):
        """Read and print stderr output from the lc0 process."""
        for line in self.process.stderr:
            print(f"lc0 stderr: {line.strip()}")

    def _send_command(self, command):
        """Send a command to the engine."""
        print(f"Sending to lc0: {command}")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _wait_for_response(self, expected_response=None):
        """Wait for a response from the engine."""
        output = []

        while True:
            # Check if there's data to read without timeout
            ready_to_read, _, _ = select.select([self.process.stdout], [], [])

            if ready_to_read:
                line = self.process.stdout.readline()
                if not line:
                    break

                line = line.strip()
                print(f"Received from lc0: {line}")
                output.append(line)

                if expected_response and expected_response in line:
                    return output

        return output

    def play(self, board, nodes=1):
        """Get a move from the engine for the given board position."""
        # Set the position
        fen = board.fen()
        self._send_command(f"position fen {fen}")

        # Ask for a move with the specified number of nodes
        self._send_command(f"go nodes {nodes}")

        # Wait for the engine to respond
        response = self._wait_for_response("bestmove")

        # Extract the move from the response
        move = None
        for line in response:
            if line.startswith("bestmove"):
                # Extract move from bestmove line
                move_match = re.search(r"bestmove\s+(\S+)", line)
                if move_match:
                    move = move_match.group(1)
                    break

        if not move:
            raise Exception("No move found in engine response")

        # Convert the move string to a chess.Move object
        try:
            chess_move = chess.Move.from_uci(move)
            return MoveResult(chess_move)
        except ValueError:
            raise Exception(f"Invalid move: {move}")

    def quit(self):
        """Quit the engine."""
        try:
            self._send_command("quit")
            self.process.wait()
        except:
            self.process.terminate()
            try:
                self.process.wait()
            except:
                self.process.kill()

class MoveResult:
    """A simple class to mimic the result object returned by chess.engine.play()."""
    def __init__(self, move):
        self.move = move
