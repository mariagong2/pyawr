'''
Time Domain Gating.

One or two port S-parameter files are supported. The gating techique uses lowpass time-domian-reflectometry
only where simulation data at DC is a requirement. If the S-parameter data does not include DC, extrapolation
will be performed.

This script works with Microwave Office Project specified by the variable AWRDE_ProjectName (see User Entry
section below). In addiiton the AWRDE version must be specified. AWRDE must be running with the specified
project opened before starting this script.

Steps:
1. In AWRDE, import the ungated s-parameter file into Data Files. This must be either .s1p or .s2p Touchstone
formatted data.
2. Run this script and choose the desired ungated s-parameter file in the prompt
3. When prompted, in AWRDE set the gating markers in the graph: TDR Lowpass Step
4. The script will perform the gating operation and place the gated file under Data Files in AWRDE with
_gated suffix attached to the original file name

Rev 1.02 9/21/2021

'''

#--------------------Do not edit below this line-------------------------------------------
import numpy as np
import tkinter as tki
from tkinter import ttk
import pyawr.mwoffice as mwo # 
import time as tm # 시간관련 함수를 제공 
import warnings # 경고메세지를 위한 모듈
import os # OS는 운영체제와 상호작용을 위한 내장 라이브러리 os.remove(Dir_n_File) 부분에서 디렉토리 삭제하기위해 있음

'''
-----tkinter VS QT Designer -----
tkinter의 장점: 
Tkinter는 Python에 기본으로 포함된 표준 GUI 라이브러리입니다(별도설치 필요없음)
가볍고 간단한 응용 프로그램을 만들 때 사용&작은 프로젝트에 적합합니다.
QT Designer의 장점:
더 복잡하고 대규모의 프로젝트에 적합
Qt Designer는 사용자 인터페이스를 시각적으로 디자인하고, Qt에 대한 코드를 생성하는 데 사용됩니다.
풍부한 위젯 라이브러리, 테마 지원, 국제화 및 기타 기능을 제공합니다.

'''

t = True
f = False
#
#******************************************************************************************
#
class class_TDR_functions():
    def __init__(self):
        pass
        
    def CalcStructureDelay(self, TDR_LPS_21_Data_ay):#---------------------------------------
        #Read s21 LPS response and determine overall delay
        #넘피로 X열과 Y열의 값을 정리해서 갖고오고 x가 0.5보다 작을때를 찾는 def        
        Time_ay = TDR_LPS_21_Data_ay[:,0] #2차원 넘피배열로,첫번째 열 데이터 갖고오는것 (이 경우 X축 데이터를 갖고옴)  
        LPS_21_ay = TDR_LPS_21_Data_ay[:,1] # 2차원 넘피배열, 두번째열의 정보를 갖고온다 (이 경우 Y축 데이터를 갖고온다)
        TraceIdx_ay = np.arange(len(Time_ay)) # len = length는 개수를 말한다 
        TraceIdx_ay = TraceIdx_ay[::-1] #윗줄에서 만든 배열을 역순으로 재배열하는것 
        for i in TraceIdx_ay:
            if LPS_21_ay[i] < 0.5: 
                break
            #end if
        #end for
        StructureDelay = Time_ay[i]
        return StructureDelay
        
    
    #TDR_Data_ay 란 시간 도메인response를 나타내는 데이터이다 
    def Truncate_TDR_Respone(self, TDR_Data_ay):
        #Remove time data < t=0
        #TDR data from MWO repeats the last 10% of the TDR response below t = 0
        # TDR특성상 시간이 0미만인경우에는 마지막 부분을 10%반복한다 
        #TDR_Data_ay format
        #   Col 0 = time arrray
        #   Col 1 = tdr_11 data, Col 2 = tdr_21 data, Col 3 = tdr_12 data, Col 4 = tdr_22 data
        # 시간이 0 미만인 데이터는 제거하고 
        
        #TDR_Data_ay format: TDR 데이터의 배열 형식에 대한 설명입니다.
        
        NumPts = len(TDR_Data_ay) #TDR_Data_ay의 길이를 가져온다 
        FirstIndex = int(-self.MinTimeBin)  # 부호에 관계없이 양수로 만들기 위한 코드..? 
        LastIndex = int(NumPts) #
        if len(TDR_Data_ay.shape) == 2: #2차원인지 파악 즉, 행렬형태인지 확인
            NumCols = len(TDR_Data_ay[0]) #첫번째행의 열의수를 numcols 라고 부름
            Truncated_TDR_Data_ay = np.zeros((NumPts,NumCols))
            # 크기가 numpts*numcols 인 영행렬을 만들기 
            Truncated_TDR_Data_ay = TDR_Data_ay[FirstIndex:LastIndex,:]
            # 영행렬에 TDR_Data_ay값을 넣는다 즉 Firstindex에서 lastIndex를 복붙
        elif len(TDR_Data_ay.shape) == 1:  
            #만약 TDR_Data_ay가 1차원이라면
            Truncated_TDR_Data_ay = np.zeros((NumPts))
            # 새로운 배열을 생성하고 0으로 초기화합니다
            Truncated_TDR_Data_ay = TDR_Data_ay[FirstIndex:LastIndex]
            # 주어진 FirstIndex부터 LastIndex까지의 범위로 TDR_Data_ay를 
            # 잘라서 Truncated_TDR_Data_ay에 할당합니다.
        #end if
        return Truncated_TDR_Data_ay
        #
        
    def GateTimeCalc(self, MarkerTime_list, TDR_Time_ay, StructureDelay, NumPorts):#---------------------------------
        #Calculate start and stop gate indecies for the TDR_Time_ay
        #주어진 시간 데이터 및 마커 시간 리스트를 이용하여 각 TDR Trace에 대한 게이트의 시작 및 종료 시간을 계산합니다
        # 시간데이터로 리스트를 정리해서게이트 시작 및 종료시간 조정한다 
        GateTimeData_list = list()
        #게이트 데이터를 저장하도록 빈 리스트를 생성한다 
        Gate_Start_time_11 = MarkerTime_list[0] #게이트 시작시간
        Gate_Stop_time_11 = MarkerTime_list[1] #게이트 종료시간 
        
        Gate_Start_idx_11 = self.FindIndex(TDR_Time_ay, Gate_Start_time_11)
        Gate_Stop_idx_11 = self.FindIndex(TDR_Time_ay, Gate_Stop_time_11)
        #TDR Trace 1에대한 게이트 시작 및 종료시간 계산 
        GateTimeData_list.append(['tdr11', Gate_Start_idx_11, Gate_Stop_idx_11, Gate_Start_time_11, Gate_Stop_time_11])
        #결과를 리스트에 추가
        
        if NumPorts == 2:   # 만약 포트의 수가 2개인 경우
            Tx_GateTimeWidth = (Gate_Stop_time_11 - Gate_Start_time_11)/2
            Gate_Start_time_21 = StructureDelay - Tx_GateTimeWidth/2
            Gate_Stop_time_21 = StructureDelay + Tx_GateTimeWidth/2
            Gate_Start_idx_21 = self.FindIndex(TDR_Time_ay, Gate_Start_time_21)
            Gate_Stop_idx_21 = self.FindIndex(TDR_Time_ay, Gate_Stop_time_21)
            #
            Gate_Start_time_22 = 2*StructureDelay - Gate_Stop_time_11
            Gate_Stop_time_22 = 2*StructureDelay - Gate_Start_time_11
            Gate_Start_idx_22 = self.FindIndex(TDR_Time_ay, Gate_Start_time_22)
            Gate_Stop_idx_22 = self.FindIndex(TDR_Time_ay, Gate_Stop_time_22)
            #
            Gate_Start_time_12 = StructureDelay - Tx_GateTimeWidth/2
            Gate_Stop_time_12 = StructureDelay + Tx_GateTimeWidth/2
            Gate_Start_idx_12 = self.FindIndex(TDR_Time_ay, Gate_Start_time_12)
            Gate_Stop_idx_12 = self.FindIndex(TDR_Time_ay, Gate_Stop_time_12)
            #계산된 게이트 데이터를 리스트에 추가한다 
            GateTimeData_list.append(['tdr21', Gate_Start_idx_21, Gate_Stop_idx_21, Gate_Start_time_21, Gate_Stop_time_21])
            GateTimeData_list.append(['tdr12', Gate_Start_idx_12, Gate_Stop_idx_12, Gate_Start_time_12, Gate_Stop_time_12])
            GateTimeData_list.append(['tdr22', Gate_Start_idx_22, Gate_Stop_idx_22, Gate_Start_time_22, Gate_Stop_time_22])
        #end if
        #
        return GateTimeData_list
        #
    def Filter_TDR_Response(self, TDR_LPI_Data_ay, GateTimeData_list, TDR_Filter_Params):#------------------------------------
        #Apply bandpass filter to TDR response
        ##TDR 응답에 대해 밴드패스 필터를 적용하는 함수
        #게이팅된 TDR 데이터에 대해 필터링 수행 단계
        
        NumTraces = len(TDR_LPI_Data_ay[0])-1 #첫번째 열은 시간데이터이며 삭제
        NumTDR_Pts = len(TDR_LPI_Data_ay[:,0]) # [:,A]라고 하면[행, 열]을 뜻한다
        # :는 전체, 즉 여기서는 TDR_LPI_Data_ay값의 모든행의 첫번째열 값의 개수를 
        #NumTDR_Pts 라는 변수값으로 넣는것이다 
        Filtered_TDR_ay = np.zeros_like(TDR_LPI_Data_ay)
        #영행렬을 만들어서 Filtered_TDR_ay라는 변수를 위해서 넣어준다 
        # like를 사용하는 이유는 TDR_LPI_Data_ay와 크기를 동일하게 하기위해서이다
        
        Filtered_TDR_ay[:,0] = TDR_LPI_Data_ay[:,0] #Time data
        
        for trace_idx in range(NumTraces):
            Temp_Unfiltered_TDR_ay = TDR_LPI_Data_ay[:,(trace_idx+1)]
            Gate_Start_idx = GateTimeData_list[trace_idx][1]
            Gate_Stop_idx = GateTimeData_list[trace_idx][2]
            Temp_TD_FilterResponse_ay = abs(self.CreateTimeFilter(NumTDR_Pts, Gate_Start_idx, Gate_Stop_idx, TDR_Filter_Params))
            Temp_Filtered_TDR_ay = Temp_TD_FilterResponse_ay * Temp_Unfiltered_TDR_ay
            Filtered_TDR_ay[:,trace_idx+1] = Temp_Filtered_TDR_ay
        #end  for
        return Filtered_TDR_ay
        #
    def CreateTimeFilter(self, NumPts, Start_idx, Stop_idx, Filter_Params):#---------------------
        FilterFunction = Filter_Params[0]
        FilterType = Filter_Params[1]
        Cuttoff_dB_Down = Filter_Params[2]
        FilterOrder = Filter_Params[3]
        #
        X_ay = np.arange(NumPts)
        Pole_list = self.FilterPolynomials(FilterFunction, Cuttoff_dB_Down, FilterOrder)
        if FilterType == 'Bandpass':
            W1 = Start_idx
            W2 = Stop_idx
            Wm = np.sqrt(W1*W2)
            BW = (W2 - W1)/Wm
            Omega_ay = X_ay/Wm
            Numerator = 1
            NumPoles = len(Pole_list)
            FreqResponse_list = list()
            for w in Omega_ay:
                Denominator = 1
                w = max(1e-20,w)
                for p_idx in range(NumPoles):
                    s = complex(0,w)
                    Denominator *= ((s**2+1)/(s*BW) - Pole_list[p_idx])
                #end for
                FreqResponse_list.append(Numerator/Denominator)
            #end for
            FilterResponse_ay = np.array(FreqResponse_list)
            #
            MaxMag = max(abs(FilterResponse_ay))
            MaxMag_idx = self.FindIndex(abs(FilterResponse_ay), MaxMag)
            FilterResponse_ay = FilterResponse_ay/FilterResponse_ay[MaxMag_idx]
            #
        elif FilterType == 'Bandstop':
            W1 = Start_idx
            W2 = Stop_idx
            Wm = np.sqrt(W1*W2)
            BW = (W2 - W1)/Wm
            Omega_ay = X_ay/Wm
            Numerator = 1
            NumPoles = len(Pole_list)
            FreqResponse_list = list()
            for w in Omega_ay:
                Denominator = 1
                w = max(1e-20,w)
                for p_idx in range(NumPoles):
                    s = complex(0,w)
                    Denominator *= ((s*BW)/(s**2+1) - Pole_list[p_idx])
                #end for
                FreqResponse_list.append(Numerator/Denominator)
            #end for
            FilterResponse_ay = np.array(FreqResponse_list)
            #
            MaxMag = max(abs(FilterResponse_ay))
            MaxMag_idx = self.FindIndex(abs(FilterResponse_ay), MaxMag)
            FilterResponse_ay = FilterResponse_ay/FilterResponse_ay[MaxMag_idx]
            #
        else:
            raise RuntimeError(FilterType+' not recognized')
        #end if
        return FilterResponse_ay
        #
    def FilterPolynomials(self, FilterFunction, Cuttoff_dB_Down, FilterOrder):#-------------------------------------
        n = FilterOrder
        Pole_list = list()
        epsilon = np.sqrt(10**(abs(Cuttoff_dB_Down)/10) - 1)
        if FilterFunction == 'Butterworth':
            epsilon_n = epsilon**(1/n)
            for k in range(n):
                a = -np.sin(np.pi*(1 + 2*k)/(2*n))
                b = np.cos(np.pi*(1 + 2*k)/(2*n))
                a /= epsilon_n
                b /= epsilon_n
                p = complex(a,b)
                Pole_list.append(p)
            #end for
        elif FilterFunction == 'Chebyshev':
            for k in range(n):
                a = -np.sin((np.pi/(2*n))*(1 + 2*k))*np.sinh((1/n)*np.arcsinh(1/epsilon))
                b = np.cos((np.pi/(2*n))*(1 + 2*k))*np.cosh((1/n)*np.arcsinh(1/epsilon))
                p = complex(a,b)
                Pole_list.append(p)
            #end for
        else:
            raise RuntimeError(FilterFunction+' not recognized')
        #end if
        return Pole_list
        #
    def Compute_FFT_Parameters(self, NumFreqPts, MaxFreq, TDR_Time_ay, TimeResolutionFactor):#-------------------------------------------
        self.Num_FreqSamplePts = NumFreqPts
        self.Fmax = MaxFreq
        self.Fstep = self.Fmax/(self.Num_FreqSamplePts-1)
        self.Tr = 1/self.Fstep
        self.Tstep = 1/(2*self.Fmax*TimeResolutionFactor)
        self.MaxTimeBin = self.Tr/self.Tstep
        self.MinTimeBin = (-1)*np.floor(self.MaxTimeBin/10)
        self.MinTime = self.MinTimeBin*self.Tstep
        self.NumTotalTimeBins = self.MaxTimeBin - self.MinTimeBin + 1
        self.Num_IFFT_Pts = int(self.NumTotalTimeBins + self.MinTimeBin)
        self.NumZeroStuffingPts = self.Num_IFFT_Pts - (2*self.Num_FreqSamplePts-2)
        #
    def Calc_Sparam_FFT(self, TDR_Data_ay, FFT_WindowType):#------------------------------------
        NumCols = len(TDR_Data_ay[0]) #열의 수를 갖고옴
        N = len(TDR_Data_ay) #데이터의 길이를 갖고옴
        if NumCols == 2: # 2열인 경우, S 매개변수는 1개
            NumSparams = 1 
        elif NumCols ==5: # 5열인 경우 S 매개변수는 4개 
            NumSparams = 4
        #end if
        SparamData_Complex_ay = np.zeros((int(self.Num_FreqSamplePts), int(NumSparams)),dtype=complex)
        # 주파수 도메인의 산란 매개변수를 저장할 배열 초기화
        
        for i in range(NumSparams): #NumSparams 수 만큼 반복 
            TimeData_ay = TDR_Data_ay[:, i + 1] # 시간 도메인 데이터 열 선택
            fft_v = np.fft.fft(TimeData_ay) * (2/N) # 
            #FFT(Fast Fourier transform)-고속푸리에변환 수행 및 스케일링
            fft_v *= self.Num_FreqSamplePts # 스케일링: 원래의 비대칭 주파수 데이터는 M 포인트를 가짐
            LastIndex = int(self.Num_FreqSamplePts)
            Truncated_fft_v = fft_v[0:LastIndex] # 거울상 부분 및 제로 패딩 제거
            Scaled_fft_v = self.UndoWindowingFromFreqData(Truncated_fft_v, FFT_WindowType) # 주파수 데이터로부터 창 제거
            Repaired_fft_v = self.ExtrapolateLastPoint(Scaled_fft_v, FFT_WindowType) # 마지막 포인트 복원
            SparamData_Complex_ay[:, i] = Repaired_fft_v # 결과를 배열에 저장
    # end for
        return SparamData_Complex_ay
        #
    def UndoWindowingFromFreqData(self, Freq_Data_ay, FFT_WindowType):#----------------------------------------
        NumFreqPts = len(Freq_Data_ay)
        NumWindowPts = 2*NumFreqPts
        n_ay = np.arange(NumWindowPts )
        if FFT_WindowType == 'Blackman':
            win_ay = 0.42 - 0.5*np.cos(2*np.pi*n_ay/NumWindowPts ) + 0.08*np.cos(4*np.pi*n_ay/NumWindowPts )
        else:
            raise RuntimeError(FFT_WindowType+' FFT Windowing Type not supported')
        #end if
        FirstIndex = int(NumWindowPts/2)
        Truncated_win_ay = abs(win_ay[FirstIndex:])
        for i in range(NumFreqPts):
            Truncated_win_ay[i] = max(Truncated_win_ay[i],1e-6)
        #end if
        Scaled_Freq_Data_ay = Freq_Data_ay/Truncated_win_ay
        return Scaled_Freq_Data_ay
        #
    def ExtrapolateLastPoint(self, fft_v, FFT_WindowType):#--------------------------------------------------
        #For some windowing types, the last point is corrupted. For these window types replace last
        # point with extrapolated from previous two point
        GoodWindowTypes = ['Rectangular', 'Hamming']
        BadWindowTypes = ['Hanning', 'Blackman', 'Bartlett', 'Lanczos']
        NeedsExtroplating = True
        for i in range(len(GoodWindowTypes)):
            if FFT_WindowType == GoodWindowTypes[i]:
                NeedsExtroplating = False
            #end if
        #end for
        for i in range(len(BadWindowTypes)):
            if FFT_WindowType == BadWindowTypes[i]:
                NeedsExtroplating = True
            #end if
        #end for
        if NeedsExtroplating:
            #Xdata is bin numbers
            y1 = fft_v[-3]
            y2 = fft_v[-2]
            y0 = 2*y2 - y1
            fft_v[-1] = y0
        #end if
        return fft_v
        #
    def ResampleSparamData(self, SparamData_Complex_ay, OldFreqParm_list, NewFreqParam_list):#-------------------------------------------------------
        #SparamData_Complex_ay is only the data in complex format
        #OldFreqParm_list coresponds to SparamData_Complex_ay
        #NewFreqParam_list corresponds to the original ungated Sparam data in Data Files
        #
        NumCols = len(SparamData_Complex_ay[0])
        OldMinFreq = OldFreqParm_list[0]
        OldMaxFreq = OldFreqParm_list[1]
        OldNumFreqs = int(OldFreqParm_list[2])
        NewMinFreq = NewFreqParam_list[0]
        NewMaxFreq = NewFreqParam_list[1]
        NewNumFreqs = int(NewFreqParam_list[2])
        Sparam_RI_list = list()
        if len(SparamData_Complex_ay) != OldNumFreqs:
            print(len(SparamData_Complex_ay))
            print(OldNumFreqs)
            raise RuntimeError('Number of SparamData_Complex_ay does not match OldNumFreqs')
        #end if
        #
        OldFreq_ay = np.linspace(OldMinFreq, OldMaxFreq, OldNumFreqs)
        NewFreq_ay = np.linspace(NewMinFreq, NewMaxFreq, NewNumFreqs)
        for nf_idx in range(int(NewNumFreqs)):
            NewFreq = NewFreq_ay[nf_idx]
            TempSparam_list = list()
            TempSparam_list.append(NewFreq/1e9) #Store Freq as GHz
            for col_idx in range(NumCols):
                y = np.interp(NewFreq, OldFreq_ay, SparamData_Complex_ay[:,col_idx])
                TempSparam_list.append(y.real)
                TempSparam_list.append(y.imag)
            #end for
            Sparam_RI_list.append(TempSparam_list)
        #end for
        Sparam_RI_ay = np.array(Sparam_RI_list)
        return Sparam_RI_ay
        #
    def FindIndex(self, DataVector, DataValue, IndexMethod='Closest Index', ClipMethod = 'NoClip'):#---------------------------------------------
        #Finds the index of a vector (DataVector)
        #IndexMethod:
        #    Closest Index
        #    Next Higher Index
        #    Next Lower Index
        #
        NumPts = self.ArraySize(DataVector)[1]
        FudgeFactor = 1.00001
        MaxDataValue = max(DataVector)
        if MaxDataValue >= 0:
            MaxDataValue = MaxDataValue*FudgeFactor
        else:
            MaxDataValue = MaxDataValue/FudgeFactor
        #end if
        MinDataValue = min(DataVector)
        if MinDataValue >= 0:
            MinDataValue = MinDataValue/FudgeFactor
        else:
            MinDataValue = MinDataValue*FudgeFactor
        #end if
        if DataValue < MinDataValue or DataValue > MaxDataValue:
            if ClipMethod == 'NoClip':
                raise RuntimeError('FindIndex: DataValue outside of DataVector range '+str(DataValue))
            elif ClipMethod == 'Clip':
                if DataValue < MinDataValue:
                    DataValue = MinDataValue
                elif DataValue > MaxDataValue:
                    DataValue = MaxDataValue
                #end if
            #end if
        #end if
        #
        FirstPass = True
        for d_idx in range(NumPts):
            if FirstPass:
                MinError = abs(DataVector[d_idx] - DataValue)
                MinErrorIdx = d_idx
                FirstPass = False
            else:
                TempMinError = abs(DataVector[d_idx] - DataValue)
                if TempMinError < MinError:
                    MinError = TempMinError
                    MinErrorIdx = d_idx
            #end if FirstPass
        #end for d_idx

        if IndexMethod == 'Closest Index':
            Index = MinErrorIdx
        elif IndexMethod == 'Next Higher Index':
            Index = MinErrorIdx + 1
            Index = min(Index,NumPts-1)
        elif IndexMethod == 'Next Lower Index':
            Index = MinErrorIdx - 1
            Index = max(MinErrorIdx, 0)
        else:
            raise RuntimeError('Invalide IndexMethod:'+IndexMethod)
        #end if
        return Index
        #
    def ArraySize(self, Array): #-----------------------------------------------------------------------
        #Return size of Array
        if type(Array) != 'numpy.darray':
            Array = np.array(Array)
        #end
        size_tuple = Array.shape
        NumDimensions = len(size_tuple)
        if NumDimensions == 1:
            Size = [1,size_tuple[0]]
        elif NumDimensions == 2:
            Size = [size_tuple[0],size_tuple[1]]
        elif NumDimensions == 3:
            Size = [size_tuple[0],size_tuple[1],size_tuple[2]]
        else:
            print('MiscFunctions.ArraySize: More than 3 is TODO')
        return Size
        #
#
#******************************************************************************************
#
class class_AWRDE_Interface(): #Set up 첫번째 부분 
    def __init__(self):#-------------------------------------------------------------------
        self.ProjectName = ''
        self.LinkEstablished = False
        self.SchemName = ''
        self.SparamMagGraphName = ''
        self.SparamPhaseGraphName = ''
        self.LPI_GraphName = ''
        self.LPS_GraphName = ''
        #클래스 초기화 및 변수 선언 
    def EstablishLink(self, awrde, ProjectName='none'):#---------------------------------------------
        self.awrde = awrde
        # 
        try:
            OpenedProjectName = self.awrde.Project.Name
        except:
            raise RuntimeError('Error in opening AWRDE project')
        #end try
        #
        if ProjectName == 'none':
            print('')
            print('AWRDE link established. Project Name: '+OpenedProjectName)
            print('')
        else: # 그외에는 
            if not '.emp' in ProjectName:
                ProjectName += '.emp'
            #end if
            if OpenedProjectName == ProjectName:
                print('')
                print('AWRDE link established. Project Name: '+OpenedProjectName)
                print('')
            else:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('Incorrect Opened Project:')
                print('   Desired Project: '+ProjectName)
                print('   Opened Project: '+OpenedProjectName)
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                raise RuntimeError('')
            #end if
        #end if
        return self.awrde
        #    
    def GetProjectPath(self):#-----------------------------------------------------------------
        ProjectDirectory = self.awrde.Project.Path
        ProjectDirectory = ProjectDirectory.replace('\\','/')
        return ProjectDirectory
        #
    def Get_TraceNames(self, GraphName):#--------------------------------------------------------
        TraceName_list = list()
        try:
            graph = self.awrde.Project.Graphs(GraphName)
        except:
            raise RuntimeError('Invalid Graph Name: '+GraphName)
        #end try
        NumTraces = graph.Measurements.Count
        for i in range(NumTraces):
            TraceName_list.append([graph.Traces[i].Name,  graph.Measurements[i].OnLeftAxis])
        #end for
        return TraceName_list
        #
    def Write_TraceNames(self, GraphName, TraceName_list):#-----------------------------------------
        #Removes all existing traces and adds new ones from TraceName_list
        NumTraces = len(TraceName_list)
        try:
            graph = self.awrde.Project.Graphs(GraphName)
        except:
            raise RuntimeError('Invalid Graph Name: '+GraphName)
        #end try
        #
        graph.Measurements.RemoveAll()  #Looks like you can't edit traces, must add new one. Easiest to delete all first
        for trace_idx in range(NumTraces):
            TraceName = TraceName_list[trace_idx][0]
            ColonLoc = TraceName.find(':')
            SourceDoc_str = TraceName[:ColonLoc]
            Measurement_str = TraceName[(ColonLoc+1):]
            graph.Measurements.Add(SourceDoc_str,Measurement_str)
            graph.Measurements[trace_idx].OnLeftAxis = TraceName_list[trace_idx][1]
        #end for
        #
    def Write_Equation(self, DocName, EquationName, Expression_str, DocType='Schematic'):#--------------
        if DocType == 'Schematic':
            schem = self.awrde.Project.Schematics(DocName)
            schem.Equations(EquationName).Expression = EquationName+' = '+Expression_str
        else:
            raise RuntimeError('DocType not supported')
        #end if
        #
    def GetTraceXdata_Ay(self, GraphName, MeasurementName):#----------------------------------------
        Xdata_list = list()
        try:
            graph = self.awrde.Project.Graphs(GraphName)
        except:
            raise RuntimeError('Invalid Graph Name: '+GraphName)
        #end try
        try:
            TraceData = graph.Measurements(MeasurementName)
        except:
            raise RuntimeError('Invalid Measurement Name: '+MeasurementName+'  Graph Name: '+GraphName)
        NumTracePts = TraceData.XPointCount
        for i in range(NumTracePts):
            Xdata_list.append(TraceData.XValue(i+1))
        #end for
        Xdata_ay = np.array(Xdata_list)
        return Xdata_ay
        #
    def GetTraceYdata_Ay(self, GraphName, MeasurementName):#----------------------------------------
        Ydata_list = list()
        try:
            graph = self.awrde.Project.Graphs(GraphName)
        except:
            raise RuntimeError('Invalid Graph Name: '+GraphName)
        #end try
        try:
            TraceData = graph.Measurements(MeasurementName)
        except:
            raise RuntimeError('Invalid Measurement Name: '+MeasurementName)
        NumTracePts = TraceData.XPointCount
        for i in range(NumTracePts):
            Ydata_list.append(TraceData.YValue(i+1,1))
        #end for
        Ydata_ay = np.array(Ydata_list)
        return Ydata_ay
        #
    def GetTraceXYdata_Ay(self, GraphName, MeasurementName, single_ay=True):#----------------------------------------
        Xdata_list = list()
        Ydata_list = list()
        try:
            graph = self.awrde.Project.Graphs(GraphName)
        except:
            raise RuntimeError('Invalid Graph Name: '+GraphName)
        #end try
        try:
            TraceData = graph.Measurements(MeasurementName)
        except:
            raise RuntimeError('Invalid Measurement Name: '+MeasurementName+'  Graph Name: '+GraphName)
        NumTracePts = TraceData.XPointCount
        for i in range(NumTracePts):
            Xdata_list.append(TraceData.XValue(i+1))
            Ydata_list.append(TraceData.YValue(i+1,1))
        #end for
        Xdata_ay = np.array(Xdata_list)
        Ydata_ay = np.array(Ydata_list)
        if single_ay:
            XYdata_ay = np.zeros((NumTracePts,2))
            XYdata_ay[:,0] = Xdata_ay
            XYdata_ay[:,1] = Ydata_ay
            return XYdata_ay
        else:
            return Xdata_ay, Ydata_ay
        #end if
        #
    def SetGraphAxes(self, GraphName, WhichAxis, Min, Max, Step=-999):
        #Set the Axis limits
        #Step=-999 for auto step
        #Min or Max = -999 to leave value as is
        if WhichAxis == 'X':
            Axis_num = 1
        elif WhichAxis == 'Y Left':
            Axis_num = 2
        elif WhichAxis == 'Y Right':
            Axis_num = 3
        else:
            raise RuntimeError(WhichAxis+' not recognized')
        #end if
        try:
            graph = self.awrde.Project.Graphs(GraphName)
        except:
            raise RuntimeError('Invalid Graph Name: '+GraphName)
        #end try
        ax = graph.Axes(Axis_num)
        if Min != -999:
            ax.MinimumScale = Min
        #end if
        if Max != -999:
            ax.MaximumScale = Max
        #end if
        if Step == -999:
            ax.MinorGridlinesAutoStep = True
        else:
            ax.MinorGridlinesStep = Step
        #end if
        #

    def LaunchSimulation(self, PingInterval=1, MaxTime=120):#-----------------------------------------------------------------------
        debug_on = f
        self.awrde.Project.Simulator.Analyze()
        self.awrde.Project.Simulator.start()
        MeasurementDone = False
        ElapsedTime = 0
        StartTime = tm.time()
        while not MeasurementDone or ElapsedTime>MaxTime:
            SimStatus = self.awrde.Project.Simulator.AnalyzeState
            if SimStatus == 3:
                MeasurementDone = True
            #end if
            tm.sleep(PingInterval)
            ElapsedTime = tm.time() - StartTime
        #end while
        if MeasurementDone:
            if debug_on:
                print('Simulation Complete')
            #end if
        else:
            raise RuntimeError('!!!!! Simulation Failed  !!!!')
        #end
        return MeasurementDone
        #
    def GetMarkerData(self, GraphName, TraceName):#-------------------------------------------------------
        #MakerData_list: [x-val, y-val, Marker Name, Trace Measuremnt name, Mesurement Enabled]
        #
        MarkerData_list = list()
        try:
            graph = self.awrde.Project.Graphs(GraphName)
        except:
            raise RuntimeError('Invalid Graph Name: '+GraphName)
        #end try
        NumMarkers = graph.Markers.Count
        for m_idx in range(NumMarkers):
            Marker = graph.Markers(m_idx+1)
            MeasurementName = Marker.Measurement
            if TraceName == MeasurementName:
                MarkerXval = Marker.DataValue(1)
                MarkerData_list.append(MarkerXval)
            #end if
        #end for
        if MarkerData_list[1] < MarkerData_list[0]:
            MinVal = MarkerData_list[1]
            MaxVal = MarkerData_list[0]
            MarkerData_list = [MinVal, MaxVal]
        #end if
        MarkerData_ay = np.array(MarkerData_list)
        return MarkerData_ay
        #
    def GetMeasurementEnabledState(self, GraphName):#----------------------------------------------------
        MeasurementState_list = list()
        try:
            graph = self.awrde.Project.Graphs(GraphName)
        except:
            raise RuntimeError('Invalid Graph Name: '+GraphName)
        #end try
        NumMeasurements = graph.Measurements.count
        for i in range(NumMeasurements):
            TraceName = graph.Measurements[i].Name
            EnabledState = graph.Measurements[i].Enabled
            MeasurementState_list.append([TraceName, EnabledState])
        #end for
        return MeasurementState_list
        #
    def SetMeasurementEnabledState(self, GraphName, MeasurementState_list):#------------------------------
        NumTraces = len(MeasurementState_list)
        try:
            graph = self.awrde.Project.Graphs(GraphName)
        except:
            raise RuntimeError('Invalid Graph Name: '+GraphName)
        #end try
        for i in range(NumTraces):
            TraceName = MeasurementState_list[i][0]
            EnabledState = MeasurementState_list[i][1]
            graph.Measurements(TraceName).Enabled = EnabledState
        #end for
        #
    def SetMarkers(self, GraphName, MarkerData_list):#----------------------------------------------------
        NumMarkers = len(MarkerData_list)
        try:
            graph = self.awrde.Project.Graphs(GraphName)
        except:
            raise RuntimeError('Invalid Graph Name: '+GraphName)
        #end try
        #
        for i in range(NumMarkers):
            MarkerXval = MarkerData_list[i][0]
            TraceNameName = MarkerData_list[i][3]
            graph.Markers.Add(TraceNameName, 1, MarkerXval)
        #end for
    def ReadDataFiles(self, FileType='Any'):
        if FileType == 'Any':
            FileTypeNum = 999
        elif FileType == 'S-parameter':
            FileTypeNum = 0
        else:
            raise RuntimeError(FileType+' not supported')
        #end if
        NumDataFiles = self.awrde.Project.DataFiles.Count
        DataFile_list = list()
        for i in range(NumDataFiles):
            if self.awrde.Project.DataFiles[i].Type == FileTypeNum:
                DataFile_list.append(self.awrde.Project.DataFiles(i+1).Name)
            elif FileType == 'Any':
                DataFile_list.append(self.awrde.Project.DataFiles(i+1).Name)
            #end if
        #end for
        return DataFile_list
        #
    def Construct_TDR_Sim_Schematic(self, SchemName, NumPorts, DataFileName, MaxFreq, Num_TDR_SimPts):#----------------------------------
        NumSchematics = self.awrde.Project.Schematics.Count
        for i in range(NumSchematics):
            if self.awrde.Project.Schematics[i].Name == SchemName:
                self.awrde.Project.Schematics.Remove(i+1)
            #end if
        #end for
        self.awrde.Project.Schematics.Add(SchemName)
        schem = self.awrde.Project.Schematics(SchemName)
        #
        schem.Elements.Add('PORT',0,0)
        schem.Elements.AddSubcircuit(DataFileName,500,0)
        schem.Wires.Add(0,0,500,0)
        if NumPorts == 2:
            schem.Elements.Add('PORT',2000,0,180)
            schem.Wires.Add(1500,0,2000,0)
        #end if
        schem.Wires.Cleanup()
        schem.Elements.Add('SWPFRQ',0,1000)
        NumElements = schem.Elements.Count
        for i in range(NumElements):
            if 'SWPFRQ' in schem.Elements[i].Name:
                elem = schem.Elements[i]
                NumParameters = elem.Parameters.Count
                for j in range(NumParameters):
                    if elem.Parameters[j].Name == 'Values':
                        param = elem.Parameters[j]
                        param.ValueAsString = 'swplin(0,'+str(MaxFreq)+','+str(Num_TDR_SimPts)+')'
                    #end if
                #end for
            #end if
        #end for
        #
    def Get_DataFileFreqRange(self, DataFileName):#-----------------------------------
        NumOutEqnsDocs = self.awrde.Project.OutputEquationDocuments.Count
        OutEqnDocName = 'zz_OutEqn'
        for i in range(NumOutEqnsDocs):
            if self.awrde.Project.OutputEquationDocuments[i].Name == OutEqnDocName:
                self.awrde.Project.OutputEquationDocuments.Remove(i+1)
            #end if
        #end for
        self.awrde.Project.OutputEquationDocuments.Add(OutEqnDocName)
        oe = self.awrde.Project.OutputEquationDocuments(OutEqnDocName)
        oe.Equations.Add('s21='+DataFileName+':|S(1,1)|',0,0)
        oe.Equations.Add('FreqAy=swpvals(s21)',0,100)
        oe.Equations.Add('MinFreq=amin(FreqAy)',0,200)
        oe.Equations.Add('MaxFreq=amax(FreqAy)',0,300)
        oe.Equations.Add('NumFreqs=vlen(FreqAy)',0,400)
        #
        GraphName = 'zz_Freqs'
        NumGraphs = self.awrde.Project.Graphs.Count
        for i in range(NumGraphs):
            if self.awrde.Project.Graphs[i].Name == GraphName:
                self.awrde.Project.Graphs.Remove(i+1)
            #end if
        #end for
        self.awrde.Project.Graphs.Add(GraphName, 4) #Tabular Graph
        gr = self.awrde.Project.Graphs(GraphName)
        gr.Measurements.Add(OutEqnDocName,'Re(Eqn(MinFreq))')
        gr.Measurements.Add(OutEqnDocName,'Re(Eqn(MaxFreq))')
        gr.Measurements.Add(OutEqnDocName,'Re(Eqn(NumFreqs))')
        self.LaunchSimulation()
        MinFreq = gr.Measurements[0].YValue(1,1)
        MaxFreq = gr.Measurements[1].YValue(1,1)
        NumFreqs = gr.Measurements[2].YValue(1,1)
        DF_Freq_list = [MinFreq, MaxFreq, NumFreqs]
        #
        NumOutEqnsDocs = self.awrde.Project.OutputEquationDocuments.Count
        for i in range(NumOutEqnsDocs):
            if self.awrde.Project.OutputEquationDocuments[i].Name == OutEqnDocName:
                self.awrde.Project.OutputEquationDocuments.Remove(i+1)
            #end if
        #end for
        #
        NumGraphs = self.awrde.Project.Graphs.Count
        for i in range(NumGraphs):
            if self.awrde.Project.Graphs[i].Name == GraphName:
                self.awrde.Project.Graphs.Remove(i+1)
            #end if
        #end for
        DF_Freq_ay = np.array(DF_Freq_list)
        return DF_Freq_ay
        #
    def Construct_Sparam_Graphs(self, NumPorts):#------------------------------------
        NumGraphs = self.awrde.Project.Graphs.Count
        for i in range(NumGraphs):
            if self.awrde.Project.Graphs[i].Name == self.SparamMagGraphName:
                self.awrde.Project.Graphs.Remove(i+1)
                break
            #end if
        #end for
        NumGraphs = self.awrde.Project.Graphs.Count
        for i in range(NumGraphs):
            if self.awrde.Project.Graphs[i].Name == self.SparamPhaseGraphName:
                self.awrde.Project.Graphs.Remove(i+1)
                break
            #end if
        #end for
        self.awrde.Project.Graphs.Add(self.SparamMagGraphName, 3) #Rectangular Graph
        self.awrde.Project.Graphs.Add(self.SparamPhaseGraphName, 3)
        #
        gr = self.awrde.Project.Graphs(self.SparamMagGraphName)
        gr.Measurements.Add(self.SchemName+'.$FSWP1','DB(|S(1,1)|)')
        if NumPorts == 2:
            gr.Measurements.Add(self.SchemName+'.$FSWP1','DB(|S(2,1)|)')
            gr.Measurements.Add(self.SchemName+'.$FSWP1','DB(|S(1,2)|)')
            gr.Measurements.Add(self.SchemName+'.$FSWP1','DB(|S(2,2)|)')
        #end if
        gr = self.awrde.Project.Graphs(self.SparamPhaseGraphName)
        gr.Measurements.Add(self.SchemName+'.$FSWP1','Ang(S(1,1))')
        if NumPorts == 2:
            gr.Measurements.Add(self.SchemName+'.$FSWP1','Ang(S(2,1))')
            gr.Measurements.Add(self.SchemName+'.$FSWP1','Ang(S(1,2))')
            gr.Measurements.Add(self.SchemName+'.$FSWP1','Ang(S(2,2))')
        #end if
        #
    def Construct_LPI_Graph(self, NumPorts, WindowType, TimeResolutionFactor):#------------------
        NumGraphs = self.awrde.Project.Graphs.Count
        for i in range(NumGraphs):
            if self.awrde.Project.Graphs[i].Name == self.LPI_GraphName:
                self.awrde.Project.Graphs.Remove(i+1)
                break
            #end if
        #end for
        self.awrde.Project.Graphs.Add(self.LPI_GraphName,3)
        #
        Window_dict = {'Rectangular':1, 'Lanczos':2, 'Bartlett':3, 'Hanning':4, 'Hamming':5, 'Blackman':6}
        WindowNum_str = str(Window_dict[WindowType])
        Port_list = ['1,1','2,1','1,2','2,2']
        #
        gr = self.awrde.Project.Graphs(self.LPI_GraphName)
        Meas_str = 'Re(TDR_LPI('+Port_list[0]+',0,'+str(TimeResolutionFactor)+','+WindowNum_str+',0))[*]'
        gr.Measurements.Add(self.SchemName+'.$FSWP1',Meas_str)
        if NumPorts == 2:
            for i in range(3):
                Meas_str = 'Re(TDR_LPI('+Port_list[i+1]+',0,'+str(TimeResolutionFactor)+','+WindowNum_str+',0))[*]'
                gr.Measurements.Add(self.SchemName+'.$FSWP1',Meas_str)
            #end for
        #end if
        #
    def Construct_LPS_Graph(self, NumPorts, WindowType, TimeResolutionFactor):#------------------
        NumGraphs = self.awrde.Project.Graphs.Count
        for i in range(NumGraphs):
            if self.awrde.Project.Graphs[i].Name == self.LPS_GraphName:
                self.awrde.Project.Graphs.Remove(i+1)
                break
            #end if
        #end for
        self.awrde.Project.Graphs.Add(self.LPS_GraphName,3)
        #
        Window_dict = {'Rectangular':1, 'Lanczos':2, 'Bartlett':3, 'Hanning':4, 'Hamming':5, 'Blackman':6}
        WindowNum_str = str(Window_dict[WindowType])
        Port_list = ['1,1','2,1','1,2','2,2']
        #
        gr = self.awrde.Project.Graphs(self.LPS_GraphName)
        gr.Legend.Visible = False
        Meas_str = 'Re(TDR_LPS('+Port_list[0]+',0,'+str(TimeResolutionFactor)+','+WindowNum_str+',0))[*]'
        gr.Measurements.Add(self.SchemName+'.$FSWP1',Meas_str)
        if NumPorts == 2:
            Meas_str = 'Re(TDR_LPS('+Port_list[1]+',0,'+str(TimeResolutionFactor)+','+WindowNum_str+',0))[*]'
            gr.Measurements.Add(self.SchemName+'.$FSWP1',Meas_str)
            meas = gr.Measurements[1]
            meas.OnLeftAxis = False
        #end if
        #
        ax = gr.Axes(2)
        ax.LabelText = 'Rho'
        ax = gr.Axes(1)
        ax.MinimumScale = -0.2
        #
    def Remove_LPS_21_Trace(self):#---------------------------------------------------------
        gr = self.awrde.Project.Graphs(self.LPS_GraphName)
        meas = gr.Measurements[1]
        gr.Measurements.Remove(meas.Name)
        #
    def Place_LPS_GraphMarkers(self, GraphName, NumPorts, StructureDelay):#------------------------------
        gr = self.awrde.Project.Graphs(GraphName)
        meas = gr.Measurements[0]
        gr.Markers.Add(meas.Name,1,0)
        if NumPorts == 1:
            gr.Markers.Add(meas.Name,1,1e-9)
        elif NumPorts == 2:
            gr.Markers.Add(meas.Name,1,StructureDelay*2)
        #end if
        #
    def RemoveGraphs(self, GraphName_list):#------------------------------------------
        NumGraphNames = len(GraphName_list)
        #
        for j in range(NumGraphNames):
            NumGraphs = self.awrde.Project.Graphs.Count
            for i in range(NumGraphs):
                if self.awrde.Project.Graphs[i].Name == GraphName_list[j]:
                    self.awrde.Project.Graphs.Remove(i+1)
                #end if
            #end for
        #end for
        #
        
        
    def RemoveSchematics(self, SchemName_list):#-----------------------------------------------
        NumSchemNames = len(SchemName_list)
        #
        for j in range(NumSchemNames):
            NumSchems = self.awrde.Project.Schematics.Count
            for i in range(NumSchems):
                if self.awrde.Project.Schematics[i].Name == SchemName_list[j]:
                    self.awrde.Project.Schematics.Remove(i+1)
                #end if
            #end for
        #end for
        #
    def ImportDataFile(self, DF_Name, Dir_n_File, FileType = 'Touchstone'):#-------------------------
        NumDataFiles = self.awrde.Project.DataFiles.Count
        for i in range(NumDataFiles):
            if self.awrde.Project.DataFiles[i].Name == DF_Name:
                self.awrde.Project.DataFiles.Remove(i+1)
                break
            #end if
        #end for
        if FileType == 'Touchstone':
            self.awrde.Project.DataFiles.Add(DF_Name, Dir_n_File, True, 0)
        #end if
        #
#
#******************************************************************************************
#
class class_UI():
    def __init__(self):#----------------------------------------------
        self.CancelPressed = False
        #
    def ok_button(self):#--------------------------------------------
        self.root.destroy()
        self.root.quit()
        #
    def cancel_button(self):#----------------------------------------------------------
        self.root.destroy()
        self.root.quit()
        self.CancelPressed = True
        #
    def GetFileChosen(self, event=None):#-------------------------------------------------
        self.DataFileSelected = self.FileChosen.get()
        #
    def GetFilterTypeChosen(self, event=None):#--------------------------------------------
        self.FilterTypeSelected = self.FilterTypeChosen.get()
        #
    def DataFile_UI(self, DataFile_list):#-------------------------
        self.root = tki.Tk()
        self.root.title('Select Data File')
        self.root.geometry('400x200+50+500')
        self.mainframe = ttk.Frame(self.root, padding='10 10 10 10', borderwidth=20, relief='ridge')
        self.mainframe.grid(column=0, row=0)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        ttk.Label(self.mainframe, text='Select Data File').grid(column=0, row=0, sticky=(tki.W, tki.E))
        self.FileChosen = ttk.Combobox(self.mainframe, values=DataFile_list, width=30)
        self.FileChosen.grid(column=0, row=1, sticky=tki.W)
        if DataFile_list:
            self.FileChosen.current(0)
        self.FileChosen.bind('<<ComboboxSelected>>', self.GetFileChosen)
        ttk.Button(self.mainframe, text='OK', command=self.ok_button).grid(column=0, row=2, sticky=(tki.W))
        ttk.Button(self.mainframe, text='Cancel', command=self.cancel_button).grid(column=0, row=2, sticky=(tki.E))
        for child in self.mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)
        #end for
        self.DataFileSelected = self.FileChosen.get()
        self.root.mainloop()
        self.root.quit()
        return self.DataFileSelected
        #
    def Markers_UI(self):#---------------------------------------------
        self.root = tki.Tk()
        self.root.title('Lowpass Step Markers')
        self.root.geometry('300x200+250+500')
        self.mainframe = ttk.Frame(self.root, padding='10 10 10 10', borderwidth=20, relief='ridge')
        self.mainframe.grid(column=0, row=0)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        ttk.Label(self.mainframe, text='Adjust the gating markers on').grid(column=0, row=0, sticky=(tki.N, tki.W, tki.E))
        ttk.Label(self.mainframe, text='the TDR Lowpass Step graph').grid(column=0, row=1, sticky=(tki.W, tki.E))
        ttk.Label(self.mainframe, text='').grid(column=0, row=2, sticky=(tki.W, tki.E))
        ttk.Label(self.mainframe, text='Select Time Domain Filter Type').grid(column=0, row=3, sticky=(tki.W, tki.E))
        FilterType_list = ['Bandpass', 'Bandstop']
        self.FilterTypeChosen = ttk.Combobox(self.mainframe, values=FilterType_list, width=30)
        self.FilterTypeChosen.grid(column=0, row=4, sticky=tki.W)
        self.FilterTypeChosen.current(0)
        self.FilterTypeChosen.bind('<<ComboboxSelected>>', self.GetFilterTypeChosen)
        ttk.Label(self.mainframe, text='').grid(column=0, row=5, sticky=(tki.W, tki.E))
        ttk.Button(self.mainframe, text='OK', command=self.ok_button).grid(column=0, row=6, sticky=(tki.W))
        ttk.Button(self.mainframe, text='Cancel', command=self.cancel_button).grid(column=0, row=6, sticky=(tki.E))
        self.FilterTypeSelected = self.FilterTypeChosen.get()
        self.root.mainloop()
        self.root.quit()
        return self.FilterTypeSelected
        #
#
#******************************************************************
#
class class_FileMethods():
    def __init__(self):#--------------------------------------
        pass
        #
    def WriteDataFile(self, Dir_n_File, Data_ay, string=False):
        fw = open(Dir_n_File, 'w+')
        if string:
            fw.write(Data_ay)
        else:
            if len(Data_ay.shape) == 2:
                NumRows = len(Data_ay)
                NumCols = len(Data_ay[0])
                for i in range(NumRows):
                    line_str = ''
                    for j in range(NumCols):
                        if j < (NumCols-1):
                            line_str += str(Data_ay[i,j])+', '
                        else:
                            line_str += str(Data_ay[i,j])+'\n'
                        #end if
                    #end for
                    fw.write(line_str)
                #end for
                #
            elif len(Data_ay.shape) == 1:
                NumCols = len(Data_ay)
                line_str = ''
                for i in range(NumCols):
                    if i < (NumCols-1):
                        line_str += str(Data_ay[i])+', '
                    else:
                        line_str += str(Data_ay[i])+'\n'
                    #end if
                #end if
                fw.write(line_str)
                #
            elif len(Data_ay.shape) == 0:
                fw.write(str(Data_ay)+'\n')
            #end
        #end if
        fw.close()
        print('File Written:')
        print('   '+Dir_n_File)
        #
    def ReadDataFile(self, Dir_n_File, string=False):
        EOF_Found = False
        ReadData_list = list()
        FirstLine = True
        #
        fr = open(Dir_n_File,'r')
        if string:
            while not EOF_Found:
                line_str = fr.readline()
                line_str = line_str.lstrip()
                line_str = line_str.rstrip()
                if line_str == '':
                    EOF_Found = True
                else:
                    ReadData_list.append(line_str)
                #end if
            #end while
        else:
            while not EOF_Found:
                line_str = fr.readline()
                line_str = line_str.lstrip()
                line_str = line_str.rstrip()
                if FirstLine:
                    FirstLine = False
                    NumCommas = line_str.count(',')
                    if NumCommas == 0:
                        NumCols = 1
                    else:
                        NumCols = NumCommas + 1
                    #end if
                #end if
                if line_str == '':
                    EOF_Found = True
                else:
                    if NumCols == 1:
                        ReadData_list.append(float(line_str))
                    else:
                        line_str_split = line_str.split(',')
                        TempLine = list()
                        for i in range(NumCols):
                            TempLine.append(float(line_str_split[i]))
                        #end for
                        ReadData_list.append(TempLine)
                    #end if
                #end if
            #end while
        #end if
        fr.close()
        Data_ay = np.array(ReadData_list)
        return Data_ay
        #
    def WriteTouchstoneFile(self, Dir_n_File, Sparam_RI_ay, NumPorts):#----------------------------------------
        if NumPorts == 1:
            Dir_n_File = Dir_n_File+'.s1p'
        else:
            Dir_n_File = Dir_n_File+'.s2p'
        #end if
        NumCols = len(Sparam_RI_ay[0])
        NumRows = len(Sparam_RI_ay)
        #
        fw = open(Dir_n_File, 'w+')
        fw.write('# GHz S RI R 50\n')
        for r_idx in range(NumRows):
            line_str = ''
            for c_idx in range(NumCols):
                if c_idx == NumCols - 1:
                    line_str = line_str + str(Sparam_RI_ay[r_idx, c_idx]) + '\n'
                else:
                    line_str = line_str + str(Sparam_RI_ay[r_idx, c_idx]) + ' '
                #end if
            #end for
            fw.write(line_str)
        #end for
        fw.close()
        return Dir_n_File
        #
    def DeleteFile(self, Dir_n_File):#----------------------------------------------------
        os.remove(Dir_n_File)
        #
#
#*******************************************************************
#
#Setup-----------------------------------------------------------------------
mwof = class_AWRDE_Interface()
ui = class_UI()
tdr = class_TDR_functions()
fm = class_FileMethods()
awrde = mwo.CMWOffice()
#
warnings.filterwarnings('ignore')
#
TestMode = f
if TestMode:
    DataFileSelected = 'TwoPort Sparam Data'
    #DataFileSelected = 'OnePort Sparam Data'
#end if
ReadWrite_Data = f
#
Num_TDR_SimPts = 512
FFT_WindowType = 'Blackman'
TimeResolutionFactor = 3
StructureDelay = np.float64(1e-9)
TDR_Filter_Params = ['Butterworth', 'Bandpass', 1, 12] #[Filter function, Filter type, dB down, filter order]
#
mwof.EstablishLink(awrde)
#
SchemName = 'zz_TDRsim'
LPI_GraphName = 'zz_LPI'
LPS_GraphName = 'TDR Lowpass Step'
#
mwof.SchemName = SchemName
mwof.LPI_GraphName = LPI_GraphName
mwof.LPS_GraphName = LPS_GraphName
#
#Select Data File from AWRDE---------------------------------------------------
DataFile_list = mwof.ReadDataFiles('S-parameter')
if not TestMode:
    DataFileSelected = ui.DataFile_UI(DataFile_list)
    if ui.CancelPressed:
        raise SystemExit
    #end if
#end if
NumPorts = mwof.awrde.Project.DataFiles(DataFileSelected).PortCount
if NumPorts > 2:
    raise RuntimeError('Number of ports in data file must be <= 2')
#end if
#
if not TestMode:
    print('Retrieving Data File Frequencies . . .')
    DF_Freq_ay = mwof.Get_DataFileFreqRange(DataFileSelected) #Get Freq Range and Number of Freqs from Ungated data file
    MaxFreq = DF_Freq_ay[1]
    #
    print('Constructing Scheamtic and TDR LPS Graph . . .')
    mwof.Construct_TDR_Sim_Schematic(SchemName, NumPorts, DataFileSelected, MaxFreq, Num_TDR_SimPts)
    mwof.Construct_LPS_Graph(NumPorts, FFT_WindowType, TimeResolutionFactor)
    #
    print('Simulating to display TDR Lowpass Step data . . .')
    LPS_TraceName_list = mwof.Get_TraceNames(LPS_GraphName)
    mwof.LaunchSimulation()
    if NumPorts == 2:
        TDR_LPS_21_Data_ay = mwof.GetTraceXYdata_Ay(LPS_GraphName, LPS_TraceName_list[1][0])
        StructureDelay = tdr.CalcStructureDelay(TDR_LPS_21_Data_ay)
        mwof.SetGraphAxes(LPS_GraphName, 'X', -999, (StructureDelay/1e-9)*2.5, -999)
        mwof.Remove_LPS_21_Trace()
    #end if
    #
    print('Placing markers on TDR Lowpass Step graph . . .')
    mwof.Place_LPS_GraphMarkers(LPS_GraphName, NumPorts, StructureDelay)
    mwof.awrde.Windows.Tile(1)
    TimeFilterTypeSelected = ui.Markers_UI()
    if ui.CancelPressed:
        raise SystemExit
    #end if
    TDR_Filter_Params[1] = TimeFilterTypeSelected
    LPS_Markers_ay = mwof.GetMarkerData(LPS_GraphName, LPS_TraceName_list[0][0])
    #
    mwof.Construct_LPI_Graph(NumPorts, FFT_WindowType, TimeResolutionFactor)
    print('Simulating to create TDR LPI data . . .')
    mwof.LaunchSimulation()
    #
    print('Reading TDR Lowpass Impulse traces . . .')
    LPI_TraceName_list = mwof.Get_TraceNames(LPI_GraphName)
    TDR_Time_ay =  mwof.GetTraceXdata_Ay(LPI_GraphName, LPI_TraceName_list[0][0])
    NumTDR_Pts = len(TDR_Time_ay)
    if NumPorts == 1:
        TDR_LPI_Data_ay = np.zeros((NumTDR_Pts,2))
        NumTraces = 1
    elif NumPorts == 2:
        TDR_LPI_Data_ay = np.zeros((NumTDR_Pts,5))
        NumTraces = 4
    #end if
    TDR_LPI_Data_ay[:,0] = TDR_Time_ay
    for i in range(NumTraces):
        LPI_TraceName = LPI_TraceName_list[i][0]
        LPI_ay = mwof.GetTraceYdata_Ay(LPI_GraphName, LPI_TraceName)
        TDR_LPI_Data_ay[:,i+1] = LPI_ay
    #end for
    mwof.RemoveGraphs([LPI_GraphName, LPS_GraphName])
    mwof.RemoveSchematics([SchemName])
    #
    if ReadWrite_Data:
        fm.WriteDataFile(DataDirPath+'DF_Freq_ay.txt',DF_Freq_ay)
        fm.WriteDataFile(DataDirPath+'TDR_LPI_Data_ay.txt', TDR_LPI_Data_ay)
        fm.WriteDataFile(DataDirPath+'LPS_Markers_ay.txt', LPS_Markers_ay)
        fm.WriteDataFile(DataDirPath+'StructureDelay.txt', StructureDelay)
        fm.WriteDataFile(DataDirPath+'FileName.txt', DataFileSelected, string=True)
    #end if
else:
    TDR_LPI_Data_ay = fm.ReadDataFile(DataDirPath+'TDR_LPI_Data_ay.txt')
    LPS_Markers_ay = fm.ReadDataFile(DataDirPath+'LPS_Markers_ay.txt')[0]
    StructureDelay = fm.ReadDataFile(DataDirPath+'StructureDelay.txt')[0]
    DF_Freq_ay = fm.ReadDataFile(DataDirPath+'DF_Freq_ay.txt')[0]
    DataFileSelected = fm.ReadDataFile(DataDirPath+'FileName.txt', string=True)[0]
    #
    TDR_Time_ay = TDR_LPI_Data_ay[:,0]
#end if
#


tdr.Compute_FFT_Parameters(Num_TDR_SimPts, MaxFreq, TDR_Time_ay, TimeResolutionFactor)
Truncated_TDR_LPI_Data_ay = tdr.Truncate_TDR_Respone(Filtered_TDR_ay) #Remove negative time data
SparamData_Complex_ay = tdr.Calc_Sparam_FFT(Truncated_TDR_LPI_Data_ay, FFT_WindowType)
#
OldFreqParm_list = [0, DF_Freq_ay[1], Num_TDR_SimPts]
Sparam_RI_ay = tdr.ResampleSparamData(SparamData_Complex_ay, OldFreqParm_list, DF_Freq_ay)
ProjectDirectory = mwof.GetProjectPath()
FileName = DataFileSelected+'_gated'
FileName = FileName.replace(' ','_')
Dir_n_File = ProjectDirectory + FileName
print('Writing gated S-paramter file: '+ FileName)
Dir_n_File = fm.WriteTouchstoneFile(Dir_n_File, Sparam_RI_ay, NumPorts)
mwof.ImportDataFile(FileName, Dir_n_File)
fm.DeleteFile(Dir_n_File)
#
#Exit----------------------------------
print('')
print('Gating calculation succeeded . . .')
