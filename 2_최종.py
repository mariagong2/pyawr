from pyawr_utils import awrde_utils    #Import pyawr_uilts module
import pyawr.mwoffice as mwo
import numpy as np
import pyasn1_modules
import pyawr_utils
import matplotlib.pyplot as plt 
 
awrde = awrde_utils.establish_link()   #Establish link between Python and AWRDE
Project = awrde_utils.Project(awrde)      #Get name of currently open project
Project.simulate_analyze()   

'프로젝트이름설정'
desired_project_name = 'pyawr_PlotMeasurementData'
awrde = mwo.CMWOffice()
awrde.Project.SaveAs(desired_project_name)
ProjectName = 'Filter 1'
schematic = awrde.Project.Schematics.Add(ProjectName)


'주파수 단위 및 배열 설정'
Freq_array = np.linspace(0, 5, 201)    
awrde.Project.Frequencies.AddMultiple(Freq_array) 
Project.set_project_frequencies(Freq_array, units_str='MHz')

'회로도 그리기'
Port1=schematic.Elements.Add("Port", x=-500, y=0)
Port2=schematic.Elements.Add("port", x=3000, y=0,RotationAngle=180)

Ind1=schematic.Elements.Add("Ind", x=0, y=0)
Ind2=schematic.Elements.Add("Ind", x=1500, y=0)

Cap1=schematic.Elements.Add("Cap", x=0, y=0,RotationAngle=-90)
Cap2=schematic.Elements.Add("Cap", x=1500, y=0,RotationAngle=-90)
Cap3=schematic.Elements.Add("Cap", x=2500, y=0,RotationAngle=-90)
L1_text="L1"
L2_text="L2"
C1_text="C1"
C2_text="C2"
C3_text="C3"

'파라미터 값 입력'
Ind1.Parameters("L").ValueAsString = L1_text
Ind2.Parameters("L").ValueAsString = L2_text
Cap1.Parameters("C").ValueAsString = C1_text
Cap2.Parameters("C").ValueAsString = C2_text
Cap3.Parameters("C").ValueAsString = C3_text


'선을 추가하기 위한 좌표 지정 및 스케매틱에 선 추가'
schematic.Wires.Add(-500, 0, 0, 0)
schematic.Wires.Add(1000, 0, 1500, 0)
schematic.Wires.Add(2500, 0, 3000, 0)

'Equation 추가 '
schematic.Equations.Add('C1= 10', 0, -1800)
schematic.Equations.Add('C2= 12.04', 0, -1600)
schematic.Equations.Add('C3= 9.011', 0, -1400)
schematic.Equations.Add('L1 = 7.149', 0, -1200)
schematic.Equations.Add('L2 = 7.149', 0, -1000)


'S-parameter 추가'
graphname = 'Filter 1 Sparam'
Project.add_rectangular_graph(graphname)  # 그래프 추가
graphs_dict = Project.graph_dict
graph = graphs_dict[graphname]
meas_dict = graph.measurements_dict
graph.add_measurement(ProjectName, measurement='DB(|S(2,1)|)')
graph.add_measurement(ProjectName, measurement='DB(|S(1,1)|)')
easurement_done = Project.simulate_analyze(ping_interval=1, max_time=120) #시뮬레이션 실행

New_C1_Value = 10

schem = awrde.Project.Schematics(ProjectName) #assign schematic object to SchematicName
NumEquations = schem.Equations.Count            #Get number of equations in schematic
for i in range(NumEquations):                   #Loop through equations
    if 'C1' in schem.Equations[i].Expression:   #Locate equation for variable C1
        schem.Equations[i].Expression = 'C1 = '+str(New_C1_Value) #Update equation value
 
#Update element value---------------------------------------------------------------
New_C2_Value = 13
elem = schem.Elements('CAP.C2')                          #assign element object to capaicitor C2
elem.Parameters('C').ValueAsDouble = New_C2_Value*1e-12  #update C2 value
# 그래프 이름
graph_name_to_check = 'Filter 1 Sparam'

# 그래프가 존재하는지 확인
graph_names = [awrde.Project.Graphs.Item(i + 1).Name for i in range(awrde.Project.Graphs.Count)]
if graph_name_to_check in graph_names:
    # 그래프 객체 가져오기
    graph = awrde.Project.Graphs(graph_name_to_check)

    # 그래프에 측정 데이터가 있는지 확인
    if graph.Measurements.Count > 0:
        # 첫 번째 측정 데이터 가져오기
        meas = graph.Measurements[1]

        # 데이터 처리
        NumTracePts = meas.XPointCount
        Xdata_ay = np.zeros((NumTracePts))
        Ydata_ay = np.zeros((NumTracePts))
        Xdata_ay = meas.XValues
        Ydata_ay = meas.YValues(1)

        # 데이터 플로팅
        plt.plot(Xdata_ay, Ydata_ay)
        plt.show()
    else:
        print(f"No measurement data found for graph '{graph_name_to_check}'.")
else:
    print(f"Graph '{graph_name_to_check}' not found.")
#Simulate---------------------------------------------------------------------------
awrde.Project.Simulator.Analyze()      #Run simulation
 
#Read Measurement Data from AWRDE Graph---------------------------------------------
graph = awrde.Project.Graphs('Filter 1 Sparam')  #assign graph object
meas = graph.Measurements[0]       
              #assign measurment object

NumTracePts = meas.XPointCount                   #Get nummber of points  in measurement data
Xdata_ay = np.zeros((NumTracePts))               #Allocate X data array
Ydata_ay = np.zeros((NumTracePts))               #Allocate Y data array
 
Xdata_ay = meas.XValues                          #Read in X trace data
Ydata_ay = meas.YValues(1)                       #Read in Y trace data
 
#Plot Data--------------------------------------------------------------------------
plt.plot(Xdata_ay, Ydata_ay)                 #Plot using matplotlib method
plt.show()
'저장'


awrde.Project.Netlists.Add('이름')


awrde.Project.Save()

