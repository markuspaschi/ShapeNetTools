# Sample .bashrc for SuSE Linux
# Copyright (c) SuSE GmbH Nuernberg

# There are 3 different types of shells in bash: the login shell, normal shell
# and interactive shell. Login shells read ~/.profile and interactive shells
# read ~/.bashrc; in our setup, /etc/profile sources ~/.bashrc - thus all
# settings made here will also take effect in a login shell.
#
# NOTE: It is recommended to make language settings in ~/.profile rather than
# here, since multilingual X sessions would not work properly if LANG is over-
# ridden in every subshell.

# Some applications read the EDITOR variable to determine your favourite text
# editor. So uncomment the line below and enter the editor of your choice :-)
#export EDITOR=/usr/bin/vim
#export EDITOR=/usr/bin/mcedit

# For some news readers it makes sense to specify the NEWSSERVER variable here
#export NEWSSERVER=your.news.server

# If you want to use a Palm device with Linux, uncomment the two lines below.
# For some (older) Palm Pilots, you might need to set a lower baud rate
# e.g. 57600 or 38400; lowest is 9600 (very slow!)
#
#export PILOTPORT=/dev/pilot
#export PILOTRATE=115200

source /forall/admin/bashrc.forall

export MIRA_PATH=/misc/svn/mira/mapa3789/wrk/mira
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/misc/svn/mira/mapa3789/wrk/mira/lib
export PATH=$PATH:/misc/svn/mira/mapa3789/wrk/mira/bin
#source /misc/svn/mira/mapa3789/wrk/mira/scripts/mirabash

export MIRA_PATH=$MIRA_PATH:/misc/svn/mira/mapa3789/wrk/mira-pkg
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/misc/svn/mira/mapa3789/wrk/mira-pkg/lib
export PATH=$PATH:/misc/svn/mira/mapa3789/wrk/mira-pkg/bin

export PATH=$PATH:/home/mapa3789/apps/eclipse

export OGRE_HOME=/usr/lib64

WRK=/misc/svn/mira/mapa3789/wrk
set WRK


ulimit -c unlimited


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/mapa3789/anaconda2/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/mapa3789/anaconda2/etc/profile.d/conda.sh" ]; then
        . "/home/mapa3789/anaconda2/etc/profile.d/conda.sh"
    else
        export PATH="/home/mapa3789/anaconda2/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

