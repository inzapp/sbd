# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

# If not running interactively, don't do anything
[ -z "$PS1" ] && return

export PS1="\[\e[31m\]sbd-docker\[\e[m\] \[\e[33m\]\w\[\e[m\] > "
export TERM=xterm-256color
alias grep="grep --color=auto"
alias ls="ls --color=auto"

echo -e "\e[1;31m"
cat<<TF

                        ..:::^^^^^:::..              ~^
                :~7J5PGBB#&@@@@@@@@@@#BBBG5J7~::^~!?5G.     ^
           :!JPB#BG5J7!?G@&G5J?7?J5G&@B?!7J5GB#&@@@5!.  .:~P!
       .!YB&BP?~:    ^P@G!:~?7!?Y?!::!P@G^    :~75B#BPG##G57    ~^
    .!P&#P7:        7@&! ?#@@#PGP5#&BJ.~#@?        :75#@#7.:~!?5G.
  ^Y&&5!.          ^@@^ G@#GG5Y#&&&&&&P :&@~           ~Y&@@@GJ!.
^P@#7.             Y@P 7#5PBPYB@@#Y5PYBY Y@P             .!B@P^
~&@#J^             Y@P 7&#P555&#B#PG&@@5 J@P             :?#@&!
 :?YGG57:          ^@&^ G@@##BG5YB&GP&B.:&@!          .!5GGY?^
      .^:           7@&! ?B#B5GBB5PGY7.~#@?           :^.
                     ^G@G!::!JY5YJ7~^!P@B~
                       ~5##G5J?7?J5G##5!
                          ^!?Y555Y?!^.

TF
echo -e "\e[0;33m"

if [[ $EUID -eq 0 ]]; then
  cat <<WARN
WARNING: You are running this container as root, which can cause new files in
mounted volumes to be created as the root user on your host machine.

To avoid this, run the container by specifying your user's userid:

$ docker run -u \$(id -u):\$(id -g) args...
WARN
else
  cat <<EXPL
You are running this container as user with ID $(id -u) and group $(id -g),
which should map to the ID and group for your user on the Docker host. Great!
EXPL
fi

# Turn off colors
echo -e "\e[m"

# activate virtual environment
source /venv/sbd/bin/activate
