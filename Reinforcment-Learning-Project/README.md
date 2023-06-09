The implementation of the RL-based movement of UAV-BS is done using NS3-AI. 

The environment which includes LTE/EPC, UEs, the UAV-BS, etc. is implemented using a c++ file in NS3, while the intelligent agent that contains Q-learning (QL), deep Q-learning (DQL), or actor-critic deep Q-learning (ACDQL) algorithm, is implemented in python. The c++ and python interact using NS3-AI module which should be installed in NS3 directory. The installation of NS3-AI should be done according to the below link:
https://github.com/hust-diangroup/ns3-ai


NS3-AI provides a shared memory to be used by both c++ and python. Below is an overall scheme. There is a shared memory with two shared structures, i.e. env and action. The env structure contains the parameters that should be calculated in c++ part (Environment part or NS3 part), while the act structure contains the parameters that should be calculated in python part. Python file puts its calculated parameters into the shared memory to be fetched by c++ file and vice versa. 
 

Regarding the UAV-BS deployment, the location information of UEs and UAV-BSs are put into the shared memory (into env) by c++ part. Then, the python part fetches the location information from the shared memory and decide the best next movement of UAV-BS. The best next movement (the corresponding action code or distance or direction) is put into the shared memory (into act) after which it is fetched by the c++ file. The c++ part then moves the UAV-BS according to the action it fetches from the shared memory. After the UAV-BS moves, the new throughput of the network (sum of data rate of users in this work) is calculated in c++ part. The reward is then calculated in c++ by comparing the old and new throughputs and is put into the shared memory (into env/reward) to be fetched by the intelligent agent in python side. The python side fetches the reward and store the experience in the form of (old_state, action, reward, new_state) to be used for its training. More details of how NS3-AI and the shared memory works is provided in below link:
https://github.com/hust-diangroup/ns3-ai


The project consists of three different algorithms, namely Q-learning (QL), deep Q-learning (DQL) and actor-critic deep Q-learning (ACDQL). In QL, the state and action space are discrete and limitted. In DQL, the state space is extended to continuous, but the action space is still discrete. in ACDQL, both state and action spaces are continuous. The ACDQL achives the best results compared to the QL and DQL. Also, the DQL algorithm achives better results compared to the QL.   


I ran this project on compute canada servers. To run the algorithms on compute canada servers, a batch file should be submitted to slurm system so that the  required resources would be allocated to the project. The batch file is provided in initial.sh which should be sent to slurm and wait for resources by running the below command:

sbatch initial.sh

Before running sbatch initial.sh, all files in ACDQL folder should be copied into the path "/scratch/b/bkantarc/nahidp/ns-allinone-3.35/ns-3.35/scratch/ACDQL".

It is similar for running QL and DQL algorithm.
