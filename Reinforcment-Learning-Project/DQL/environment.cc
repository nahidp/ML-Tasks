//////////Import necessary modules//////////////
#include "ns3/ns3-ai-module.h"
#include "ns3/log.h"
#include "ns3/core-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/config-store-module.h"
#include "ns3/lte-module.h"
#include "ns3/netanim-module.h"
#include "ns3/flow-monitor.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/ipv4-flow-classifier.h"
#include "ns3/simple-device-energy-model.h"
#include "ns3/li-ion-energy-source.h"
#include "ns3/energy-source-container.h"
#include <chrono>
#include <array>
/////////////////////////////////////////////////


using namespace ns3;
using namespace std;

///////////////// set the parammeters of Env and Act the same as python part//////////////////////////////////////
//defining structure location to represent the 2D location of users and uav-bs 
struct location           
{
float x;
float y;
}Packed;


//define structure statistics to represent throughput (data rate), packet loss, delay and reward 
struct statistics         
{
float current_reward;
float current_throughput;
float currrent_delay;
float current_pl;
}Packed;

//define Env structure that contains all the information that should be put into the shared memory by c++ to be fetched by python part
struct Env
{
location uav_location;
location ue_location[50];
statistics stat;
}Packed;     //using "packed" to stop alignment by OS

//define Act structure that contains all the information that should be put into the shared memory by python part to be fetched by c++
struct Act               
{
int action_code;
}Packed;      //using "packed" to stop alignment by OS



//////////////////////////set the required parameters of our simulation//////////////////////////////////////
uint16_t seed = 3;      
uint16_t run = 3;
uint16_t ueNum = 50;
uint16_t enbNum = 1;
uint16_t area = 1000;        //square area of interest is 1000*1000 meters
uint16_t uavHeight = 100;
uint16_t uavSpeed = 6;
uint16_t userSpeedMin = 2;
uint16_t userSpeedMax = 8;
uint16_t port = 20000;
NodeContainer ueNodes;
NodeContainer enbNodes;
double improvement;
vector<float> txPackets_old;
vector<float> rxPackets_old;
float percentage_received_old = 0;
float old_dist = 0;

Ptr<LteHelper> lteHelper;
NetDeviceContainer enbLteDevs;
NetDeviceContainer ueLteDevs;
Ptr<LteUeNetDevice> ueLteDevice;
Ptr<LteEnbNetDevice> enbDevice;
Ptr<EpcTft> tft; 

MobilityHelper enbMobility; 
Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();

Time intervalRl = MilliSeconds(10);
float rx_previous = 0;
float delay_previous = 0;
float pl_previous = 0;
float rxPackets_previous = 0;

std::ofstream traceFile;
NodeContainer remoteHostNode;
Ptr<Node> remoteHost;
Ipv4InterfaceContainer ueIpInterface;
bool firstAttach[50] = {0};

Time simTime = MilliSeconds (2000);          //the simulation time is 200 time steps of 10ms

//CourseChange() function keeps the track of moving objects//////////////
void
CourseChange (std::string context, Ptr<const MobilityModel> model)
{

Vector position = model->GetPosition ();


NS_LOG_UNCOND ("At time "<< Simulator::Now().GetSeconds()*1000<<" node ID ="<<context <<
" x = " << position.x << ", y = " << position.y << " z = " << position.z );

}


//find the current location of the UAV-BS
location GetUAVPosition (Ptr <ConstantVelocityMobilityModel> Uavmob)       
{
double x = Uavmob->GetPosition().x;
double y = Uavmob->GetPosition().y;
location uav_location;
uav_location.x = x;
uav_location.y = y;
return uav_location;
}

//find the current location of a UE
location GetUePosition (Ptr <GaussMarkovMobilityModel> Uemob)     
{
double x = Uemob->GetPosition().x;
double y = Uemob->GetPosition().y;
location ue_location;
ue_location.x = x;
ue_location.y = y;
return ue_location;
}


//Change the location of UAV-BS accoring to the fetched action from python part
bool SetUavPosition(Ptr <ConstantVelocityMobilityModel> Uavmob, int action_code)    
{
bool OutofBound_flag = false;      //a flag to show if UAV-BS location is out of boundary to later gives the agent a big negative reward
Vector m_position = Uavmob->GetPosition();
m_position.z = uavHeight;

	
// action_code=0 --> move the uav-bs to the positive direction of x axis
// action_code=1 --> move the uav-bs to the negative direction of x axis
// action_code=2 --> move the uav-bs to the negative direction of y axis
// action_code=3 --> move the uav-bs to the positive direction of y axis
// action_code=4 --> keep the uav-bs in the same location
//the speed of UAV-BS is 5 m/s
if (action_code==0)             
{
if (m_position.x+5 <= area )    
  {
      m_position.x = m_position.x + 5;  
  }
else 
{
  OutofBound_flag = true;      //if the action code want to move the uav-bs outside the boundary, we set a flag to true to later give a big penalty
	                       //at the same time, we dont move the uav-bs outside the boundary but we penalize it
}
}
else if (action_code == 1)
{
if (m_position.x-5 >= 0 )
  {
      m_position.x = m_position.x - 5;
  }
else 
{
  OutofBound_flag = true;
}
}
else if (action_code == 2)
{
if(m_position.y-5 >= 0)
  {
      m_position.y = m_position.y - 5;
  }
else 
{
  OutofBound_flag = true;
}
}
else if (action_code == 3)
{
if (m_position.y + 5 <= area)
  {
      m_position.y = m_position.y + 5;
  }
else 
{
  OutofBound_flag = true;
}
}

Uavmob->SetPosition(m_position); 
//NS_LOG_UNCOND("In NS3 ,"<<"Time: "<<Simulator::Now().GetSeconds()*1000<<",   "<<"UAV New location"<<Uavmob->GetPosition().x<<",  "<<Uavmob->GetPosition().y);

return OutofBound_flag;

}





statistics CalculateStatistics (FlowMonitorHelper *flowmon, Ptr<FlowMonitor> monitor)
{

statistics stat;
float reward = 0;
float percentage_received_current = 0;


vector<float> txPackets_interval;
vector<float> rxPackets_interval;


txPackets_interval.resize(ueNum);
rxPackets_interval.resize(ueNum);



int flow;
////////////Calculating the trhoughput and reward///////////////////////////
std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats ();
Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmon->GetClassifier ());
float rx_new = 0;
for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator iter = stats.begin (); iter != stats.end (); ++iter)
{
Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (iter->first);
if (t.sourceAddress == Ipv4Address("1.0.0.2"))           //if the sender is the defined remote host
{

  flow = iter->first -1;
  txPackets_interval[flow] = iter->second.txPackets - txPackets_old[flow];
  rxPackets_interval[flow] = iter->second.rxPackets - rxPackets_old[flow];
  txPackets_old[flow] = iter->second.txPackets ;
  rxPackets_old[flow] = iter->second.rxPackets;

  percentage_received_current += rxPackets_interval[flow] / txPackets_interval[flow];
  rx_new += iter->second.rxBytes * 8;


  //////////////////////stat to check//////////////////////////////
  NS_LOG_UNCOND("timeFirstTxPacket: "<<iter->second.timeFirstTxPacket.GetSeconds());
  NS_LOG_UNCOND("timeFirstRxPacket: "<<iter->second.timeFirstRxPacket.GetSeconds());
  NS_LOG_UNCOND("timeLastTxPacket: "<<iter->second.timeLastTxPacket.GetSeconds());
  NS_LOG_UNCOND("timeLastRxPacket: "<<iter->second.timeLastRxPacket.GetSeconds());
  NS_LOG_UNCOND("txBytes: "<<iter->second.txBytes);
  NS_LOG_UNCOND("rxBytes: "<<iter->second.rxBytes);
  NS_LOG_UNCOND("txPackets: "<<iter->second.txPackets);
  NS_LOG_UNCOND("rxPackets: "<<iter->second.rxPackets);
  NS_LOG_UNCOND("lostPackets: "<<iter->second.lostPackets);
  //////////////////////stat to check//////////////////////////////

}
}

percentage_received_current = percentage_received_current/ueNum;
improvement = percentage_received_current - percentage_received_old;      //check if the received data has increased compared to the previous time step

float rx_current = rx_new - rx_previous;
rx_previous = rx_new;
percentage_received_old =  percentage_received_current;


//give the reward or penalty based on the improvement of received data
if (reward==0)
{
if (improvement>0)
{
  reward = 1;
}
else if (improvement<0)
{
  reward = -1;
}
else if (improvement==0)
{
  reward = -0.2;
}
}

//stat.current_throughput = percentage_received_current;
stat.current_throughput = rx_current;
stat.current_reward = reward;
return stat;   //it shows if throughput has improved or worsen
}


//define class UAV_Move_RL to share a memory containing Env and Act between c++ and python
////////////////////// Connecting the environment (cpp part) with python (agent) part
class UAV_Move_RL : public Ns3AIRL<Env, Act>       //our defined class inhertis from Ns3AIRL. Refer to https://github.com/hust-diangroup/ns3-ai for more info
{
public:
UAV_Move_RL(uint16_t id);       //to manage the access of python and c++ to the shared memory
void AgentMovement(bool OutofBound_flag, Ptr<ConstantVelocityMobilityModel> mob_enb, FlowMonitorHelper *flowmon, Ptr<FlowMonitor> monitor);
void SetUavInitialState ();    
};

UAV_Move_RL::UAV_Move_RL(uint16_t id) : Ns3AIRL<Env, Act>(id) {
SetCond(2, 0);      //< Set the operation lock (even for ns-3 and odd for python).
}


//put the initial location of UAV-BS and UEs into the shared memory and fetch the first movement of the UAV-BS (identified by python) from the shared memory
void UAV_Move_RL::SetUavInitialState ()
{   
auto env = EnvSetterCond();
Ptr<ConstantVelocityMobilityModel> mob_enb = enbNodes.Get(0)->GetObject<ConstantVelocityMobilityModel>();
location uav_initial_location = GetUAVPosition(mob_enb);    //step 1 of my algorithm
env->uav_location = uav_initial_location;
//NS_LOG_UNCOND("In NS3 ,"<<" Time: "<<Simulator::Now().GetSeconds()*1000<<",   "<<"UAV initial location"<<uav_initial_location.x<<",  "<<uav_initial_location.y);

Ptr<GaussMarkovMobilityModel> mob_ue[50];
location ue_initial_location [50];
for (uint32_t i = 0; i < ueNum; ++i)
{
  mob_ue[i] = ueNodes.Get(i)->GetObject<GaussMarkovMobilityModel>();
  ue_initial_location[i] = GetUePosition(mob_ue[i]);    
  env->ue_location[i] = ue_initial_location[i];
}
SetCompleted();                                            

////////////////////////////////Fetch Initial action////////////////////////////
auto act = ActionGetterCond();            
int action_code = act->action_code;
//NS_LOG_UNCOND("In NS3 ,"<<"Time: "<<Simulator::Now().GetSeconds()*1000<<",   "<<"Fetched initial action: "<<action_code);
GetCompleted();    
SetUavPosition(mob_enb,action_code);

////////////////////////////////Fetch Initial action////////////////////////////

}


//main func for the movement of the UAV-BS
void UAV_Move_RL::AgentMovement(bool OutofBound_flag, Ptr<ConstantVelocityMobilityModel> mob_enb, FlowMonitorHelper *flowmon, Ptr<FlowMonitor> monitor)
{

//calculate the system throughput 
auto env = EnvSetterCond();    

statistics stat_new = CalculateStatistics(flowmon, monitor);   //call CalculateStatistics() function to obtain the new throughut and new reward
//NS_LOG_UNCOND("In NS3 ,"<<"Time: "<<Simulator::Now().GetSeconds()*1000<<",   "<<"Reward calculated: "<<stat_new.current_reward);

if(OutofBound_flag)      //update the reward and give a big negative reward if the OutofBound_flag flag is 1
{
stat_new.current_reward = -1000;
}

	
//////////obtain the new location of UAV and UEs to put into the shared memory
location uav_location = GetUAVPosition(mob_enb);    
// NS_LOG_UNCOND("In NS3 ,"<<"Time: "<<Simulator::Now().GetSeconds()*100<<",   "<<"UAV location"<<uav_location.x<<",  "<<uav_location.y);

Ptr<GaussMarkovMobilityModel> mob_ue[50];
location ue_new_location [50];

for (uint32_t i = 0; i < ueNum; ++i)
{
mob_ue[i] = ueNodes.Get(i)->GetObject<GaussMarkovMobilityModel>();
ue_new_location[i] = GetUePosition(mob_ue[i]);   
env->ue_location[i] = ue_new_location[i];
}
env->uav_location = uav_location;
env->stat = stat_new;
SetCompleted();             
///////////////////////////////////////////////////////////////////////////////


////////fetch the next action from the shared memory
auto act = ActionGetterCond();            
int action_code = act->action_code;
//NS_LOG_UNCOND("In NS3 ,"<<"Time: "<<Simulator::Now().GetSeconds()*1000<<",   "<<"Fetched action: "<<action_code);
GetCompleted();    
////////////////////////////////////////////////////


/////move the UAV-BS based on the fetched action
OutofBound_flag = SetUavPosition(mob_enb,action_code);
////////////////////////////////////////////////
	

/////self call AgentMovement() function every intervalRl seconds
Simulator::Schedule (intervalRl, &UAV_Move_RL::AgentMovement, this, OutofBound_flag, mob_enb, flowmon, monitor);
////////////////////////////////////////////////////////////////
}



//////define the LTE-EPC network
//refer to https://www.nsnam.org/docs/models/html/lte-design.html#epc-model for more info
int
main (int argc, char *argv[])
{

CommandLine cmd (__FILE__);
cmd.AddValue ("numUe", "Number of eNodeBs", ueNum);
cmd.AddValue ("numEnb", "Number of eNodeBs", enbNum);

cmd.AddValue ("seed", "Seed Number", seed);
cmd.AddValue ("run", "Run Number", run);

cmd.Parse (argc, argv);

Config::SetDefault ("ns3::LteEnbRrc::SrsPeriodicity", UintegerValue (320)); 
Config::SetDefault ("ns3::LteEnbMac::PreambleTransMax",UintegerValue(100));
//Config::SetDefault ("ns3::LteUePhy::EnableRlfDetection", BooleanValue (false));

//Config::SetDefault ("ns3::LteSpectrumPhy::CtrlErrorModelEnabled", BooleanValue (false));
//Config::SetDefault ("ns3::LteSpectrumPhy::DataErrorModelEnabled", BooleanValue (true));
//Config::SetDefault ("ns3::RrFfMacScheduler::HarqEnabled", BooleanValue (true));
Config::SetDefault ("ns3::LteHelper::PathlossModel", StringValue ("ns3::ItuR1411LosPropagationLossModel"));
//Config::SetDefault ("ns3::LteHelper::FfrAlgorithm", StringValue ("ns3::LteFrHardAlgorithm"));
Config::SetDefault ("ns3::LteEnbNetDevice::UlBandwidth", UintegerValue(100)); //20MHz bandwidth
Config::SetDefault ("ns3::LteEnbNetDevice::DlBandwidth", UintegerValue(100)); //20MHz bandwidth
//Config::SetDefault ("ns3::LteAmc::AmcModel", EnumValue (LteAmc::PiroEW2010));
Config::SetDefault ("ns3::LteEnbPhy::TxPower", DoubleValue (15));

traceFile.open ("traceFile.txt", std::ios_base::app);


SeedManager::SetSeed (seed);
SeedManager::SetRun (run);

lteHelper = CreateObject<LteHelper> ();
Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper> ();
//lteHelper->SetSchedulerType ("ns3::PfFfMacScheduler");
lteHelper->SetEpcHelper (epcHelper);
Ptr<Node> pgw = epcHelper->GetPgwNode ();
Ptr<Node> sgw = epcHelper->GetSgwNode ();



////////////////Remote host creation and connecting it to PGW////////////////////////////////////
remoteHostNode.Create (1);
remoteHost = remoteHostNode.Get (0);
InternetStackHelper stack;
stack.Install (remoteHostNode);
PointToPointHelper p2p;
p2p.SetDeviceAttribute ("DataRate", DataRateValue (DataRate ("10Gb/s")));
p2p.SetDeviceAttribute ("Mtu", UintegerValue (5000));
//p2p.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (10)));
NetDeviceContainer device = p2p.Install (pgw,remoteHost);
Ipv4AddressHelper internet;
internet.SetBase ("1.0.0.0", "255.0.0.0");
Ipv4InterfaceContainer ipInterface= internet.Assign (device);


Ipv4StaticRoutingHelper ipv4RoutingHelper;
Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting (remoteHost->GetObject<Ipv4> ());
remoteHostStaticRouting->AddNetworkRouteTo (Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.0.0.0"), 1);


enbNodes.Create (enbNum);
ueNodes.Create (ueNum);


//////define the mobility of UAV-BS and set it to ConstantVelocityMobilityModel and set velocity to 0 so that the positioning will be determined by python
Ptr<ListPositionAllocator> initialPositionAlloc = CreateObject<ListPositionAllocator> ();
initialPositionAlloc->Add(Vector(0, 0, uavHeight));// node
enbMobility.SetPositionAllocator(initialPositionAlloc);
enbMobility.SetMobilityModel ("ns3::ConstantVelocityMobilityModel");


enbMobility.Install (enbNodes);
//enbMobility.Install (enbNodes);


//define the Mobility of UEs according to Gauss Markov mobility model
MobilityHelper uemobility;

uemobility.SetPositionAllocator ("ns3::RandomBoxPositionAllocator",
				"X" , StringValue ("ns3::UniformRandomVariable[Min=400|Max=600]"),
				"Y" , StringValue ("ns3::UniformRandomVariable[Min=400|Max=600]"),
				"Z" , StringValue ("ns3::UniformRandomVariable[Min=1|Max=1]"));
/*
uemobility.SetMobilityModel ("ns3::RandomWalk2dMobilityModel",
"Bounds",RectangleValue (Rectangle (0, area, 0, area)),
"Speed" , StringValue ("ns3::UniformRandomVariable[Min=600|Max=800]"),
"Direction", StringValue ("ns3::UniformRandomVariable[Min=0|Max=6.28]"),
"Mode",StringValue ("Time"),
"Time", TimeValue(MilliSeconds(10.0)) );    //0~2pi
*/

uemobility.SetMobilityModel ("ns3::GaussMarkovMobilityModel",
"Bounds", BoxValue (Box (300, 700, 300, 700, 1, 1)),
"MeanVelocity", StringValue ("ns3::UniformRandomVariable[Min=600|Max=800]"),
"MeanDirection", StringValue ("ns3::UniformRandomVariable[Min=0|Max=6.28]"),
"TimeStep", TimeValue(MilliSeconds(10.0)) ,   //0~2pi
"NormalVelocity", StringValue ("ns3::NormalRandomVariable[Mean=0.0|Variance=0.0|Bound=0.0]"),
"NormalDirection", StringValue ("ns3::NormalRandomVariable[Mean=0.0|Variance=0.0|Bound=0.0]"),
"NormalPitch", StringValue ("ns3::NormalRandomVariable[Mean=0.0|Variance=0.0|Bound=0.0]") );


uemobility.Install (ueNodes);

lteHelper->SetEnbAntennaModelType ("ns3::IsotropicAntennaModel");
lteHelper->SetUeAntennaModelType ("ns3::IsotropicAntennaModel");
Config::SetDefault ("ns3::LteEnbNetDevice::DlEarfcn", UintegerValue(100));   //downlink frequency 2GHz
Config::SetDefault ("ns3::LteUeNetDevice::DlEarfcn", UintegerValue(100));


////////For reuseing frequencies in order to not have interference between eNBs//////////////////

lteHelper->SetFfrAlgorithmType ("ns3::LteFrHardAlgorithm");

for (uint32_t e = 0; e < enbNum; ++e)

{
lteHelper->SetFfrAlgorithmAttribute ("DlSubBandOffset", UintegerValue (60*e));
lteHelper->SetFfrAlgorithmAttribute ("DlSubBandwidth", UintegerValue (60*(e+1)));
enbLteDevs.Add(lteHelper->InstallEnbDevice (enbNodes.Get(e)));
}

//enbLteDevs = lteHelper->InstallEnbDevice (enbNodes);
ueLteDevs = lteHelper->InstallUeDevice (ueNodes); 

lteHelper->AddX2Interface (enbNodes);


stack.Install(ueNodes);
ueIpInterface = epcHelper->AssignUeIpv4Address (ueLteDevs);

for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
{
  Ptr<Node> ueNode = ueNodes.Get (u);
  // Set the default gateway for the UE
  Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting (ueNode->GetObject<Ipv4> ());
  ueStaticRouting->SetDefaultRoute (epcHelper->GetUeDefaultGatewayAddress (), 1);

}




tft = Create<EpcTft> ();
EpcTft::PacketFilter pf;
pf.localPortStart = 1234;
pf.localPortEnd = 1234;
tft->Add (pf);

lteHelper->ActivateDedicatedEpsBearer (ueLteDevs,
				   EpsBearer (EpsBearer::GBR_CONV_VOICE),
				   tft);                              

for (uint32_t u = 0; u < ueNum; ++u)
{
lteHelper->Attach (ueLteDevs.Get(u), enbLteDevs.Get(0));

}

rxPackets_old.resize(ueNum);
txPackets_old.resize(ueNum);

ApplicationContainer sinkApp;
ApplicationContainer clientApp;

	
//////define the UEs as receivers (sinks)
for (uint32_t ueId = 0; ueId < ueNum; ++ueId)
{
 PacketSinkHelper sink ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), port));
 //Sink=UEs
 sinkApp.Add (sink.Install (ueNodes.Get(ueId)));
////////////////////////////////////////////
	

//////define remote host as sender (client)
 OnOffHelper client ("ns3::UdpSocketFactory", InetSocketAddress (ueIpInterface.GetAddress (ueId), port));
 //client.SetConstantRate (DataRate ("1Mb/s"), 1420);
 //client.SetAttribute ("StartTime", TimeValue (Seconds (0)));
 client.SetAttribute ("StopTime", TimeValue (simTime));
 client.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=0.009]" ));
 client.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.001]" ));
 clientApp.Add (client.Install (remoteHostNode)); 
////////////////////////////////////////////
}

////////start clients and servers
sinkApp.Start (MilliSeconds (0));
clientApp.Start (MilliSeconds (0));

	
//////define flowmonitor on UEs and remotehost for network stat calculation
FlowMonitorHelper flowmon;
Ptr<FlowMonitor> monitor;
monitor = flowmon.Install (ueNodes);
monitor = flowmon.Install (remoteHost);
////////////////////////////////////////////////////////////////////////////
	
Simulator::Stop (simTime);



int memblock_key = 2333;       //the memoty block key for the dhared memory to be shared with python
UAV_Move_RL uav_Move_RL(memblock_key);   //define an object of class UAV_Move_RL  

Ptr<ConstantVelocityMobilityModel> mob_enb = enbNodes.Get(0)->GetObject<ConstantVelocityMobilityModel>();
Simulator::ScheduleNow(&UAV_Move_RL::SetUavInitialState, &uav_Move_RL);   //send the initial location of nodes to the shared memory
Simulator::Schedule(intervalRl,&UAV_Move_RL::AgentMovement, &uav_Move_RL, false, mob_enb, &flowmon, monitor);  //move the UAV-BS and calculate stat after intervalRl seconds  


std::ostringstream oss;   
////////trace sink/////////////////// 


Simulator::Run ();

//////saving the xml to test-monitor-GM.xml to be used for stat calculation in flow monitor
monitor->SerializeToXmlFile ("test-monitor-GM.xml" , true, true );


Simulator::Destroy ();

return 0;
}

