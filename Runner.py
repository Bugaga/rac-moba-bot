import sys
import os
from subprocess import Popen

from MyStrategy import MyStrategy
from RemoteProcessClient import RemoteProcessClient
from model.Move import Move


class Runner:
    def __init__(self):
        if sys.argv.__len__() == 4:
            self.remote_process_client = RemoteProcessClient(sys.argv[1], int(sys.argv[2]))
            self.token = sys.argv[3]
        else:
            runner_dir = cwd=os.path.join(os.getcwd(), "local-runner-ru")
            self.runner = Popen([os.path.join(runner_dir, "local-runner.bat")], cwd=runner_dir)
            self.remote_process_client = RemoteProcessClient("127.0.0.1", 31001)
            self.token = "0000000000000000"

    def run(self):
        try:
            self.remote_process_client.write_token_message(self.token)
            self.remote_process_client.write_protocol_version_message()
            team_size = self.remote_process_client.read_team_size_message()
            game = self.remote_process_client.read_game_context_message()

            strategies = []

            for _ in range(team_size):
                strategies.append(MyStrategy())

            while True:
                player_context = self.remote_process_client.read_player_context_message()
                if player_context is None:
                    break

                player_wizards = player_context.wizards
                if player_wizards is None or player_wizards.__len__() != team_size:
                    break

                moves = []

                for wizard_index in range(team_size):
                    player_wizard = player_wizards[wizard_index]

                    move = Move()
                    moves.append(move)
                    strategies[wizard_index].move(player_wizard, player_context.world, game, move)

                self.remote_process_client.write_moves_message(moves)
        finally:
            self.remote_process_client.close()
            self.runner.terminate()


Runner().run()
f = open('local-runner-ru/result.txt')
lines = f.readlines()
print(lines)
result = int(lines[2].split(' ')[1])
print ('Result: %d' % result)
