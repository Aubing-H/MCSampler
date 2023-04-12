# Minecraft Sampler

This is a project about sampling data mannually

## Sampling

### Keybard Listening

**Environment**

 - install minedojo: https://docs.minedojo.org/sections/getting_started/install.html
    
    i recommand git clone its code and python setup.py install firstly, than build gradle (Malmo env): enter into dir `./minedojo/sim/Malmo/Minecraft` and run ./launchClinet.sh (after build it will show a window, close it and it will show success).
 - create dir ./output/lmdb-test and ./output/video-sample or it may go wrong.
 - `python run.py [goal]`, [goal] can be any of "log, sheep, cow, pig", for example `python run.py sheep` means get sheep task

**Run and play minecraft**

 - `W/S/A/D`: to move forward/backward/left/right like general game settings.
 - `Space-key`: jump, so the agent can jump over obstacles.
 - `Shift-key`: sneak.
 - `mouse_right_on + move up/down`: adjust vertical view angle.   
 - `mouse_right_on + move left/right`: adjust horizontal view angle. 
 - `mouse_left_on`: attack.    
 - `U/O/C/E/P/X`: use/drop/craft/equip/place/destroy.
 - `0-9`: increase para of equip/place/destroy, pressing 0 will increase 10.
 - `mouse_right_on + 0-9`: increase para of craft, pressing 0 will increase 10.
 - `ESC`: exit the game.

 **How to set craft parameters and equip/place/destroy parameters**

 [Carft parameters (craft-para)]. The initial craft-para is 0. ``Hold mouse right on and click`` number 1-9, the carft-para will add 1-9, if click 0, para will add 10. When craft action is executed, the craft-para will reset to 0.

 [equip/place/destroy parameters (epd-para)]. The initial epd-para is 0. ``directly click`` number 1-9, the epd-para will add 1-9, if click 0, para will add 10. When equip/place/destroy action is executed, the epd-para will reset to 0.

 How to choose parameter number? For equip/place/destroy para, the target object is listed in the bottom grids, and it is marked with 0, 1, 2, .... by default. So pressing a number means selecting the corresponding object. For craft para, press `K` and it will show craft number available in shell window, choose one of them.
 
**Other Control Keys**

The following info will show in shell window.

 - `G`: show rgb frame data.
 - `V`: show voxels (the env show config with `use_voxels=True`).
 - `Q`: show equipment.
 - `I`: show inventory.
 - `M`: show location_stats.
 - `K`: show craft parameters availble.


