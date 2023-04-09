# Minecraft Sampler

This is a project about sample data artifically

## sample

### keybard listen

 - install minedojo: https://docs.minedojo.org/sections/getting_started/install.html
 - python run.py

``Run and play minecraft``

 - W/S/A/D: to move forward/backward/left/right like general game settings.
 - Space-key: jump, so the agent can jump over obstacles
 - Shift-key: sneak, 
 - act[3]=0-24, pitch, vertical:     mouse_right_on + move up/down
 - act[4]=0-24, yaw, horizontal:     mouse_right_on + move left/right
 - act[5]=1/2/3/4/5/6/7, use/drop/attack/craft/equip/place/destroy: u/o/mouse_left click/c/e/p/x
 - act[6], set number of craft para, click numbers of 0-9 and click c
 - act[7], set number of equip/place/destroy, mouse_right_on + click number of 0-9 and click e/p/x

 ``How to set craft parameters and equip/place/destroy parameters``

 [Carft parameters (craft-para)]. The initial craft-para is 0. ``Hold mouse right on and click`` number 1-9, the carft-para will add 1-9, if click 0, para will add 10. When craft action is executed, the craft-para will reset to 0.

 [equip/place/destroy parameters (epd-para)]. The initial epd-para is 0. ``directly click`` number 1-9, the epd-para will add 1-9, if click 0, para will add 10. When equip/place/destroy action is executed, the epd-para will reset to 0.
 




     'rgb': 'g',
    'voxels': 'v', 
    'equipment': 'q',
    'inventory': 'i',
    'compass': 'm',
    'gps': 'l',
    'biome': 'b',

``How to use key``

 Control info from obs

 - g: rgb
 - v: voxels
 - q: equipment
 - i: inventory
 - m: compass
 - l: gps
 - b: biome

