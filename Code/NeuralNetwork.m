clear all;
close all;
clc;

load train.mat
load test.mat

%Train Data
trainData=train.X;
trainData=trainData(1:784,:);
trainDataLabels=train.y;
[inputsTrain, numOfTrainSamples]=size(trainData);
numOfClasses=length(unique(trainDataLabels));

vecTrainOutputs=zeros(numOfClasses,numOfTrainSamples);

for i=1:numOfTrainSamples
    vecTrainOutputs(trainDataLabels(1,i),i)=1;
end

%Test Data
testData=test.X;
testDataLabels=test.y;
testData=testData(1:784,:);
[inputsTest, numOfTestSamples]=size(testData);

vecTestOutputs=zeros(numOfClasses,numOfTestSamples);

for i=1:numOfTestSamples
    vecTestOutputs(testDataLabels(1,i),i)=1;
end

eta=0.2;


%The Sigmoid Function
sig = @(x) 1.0 ./ ( 1.0 + exp(-x) );

%The Softmax Function
%soft = @(x) exp(x)/sum(exp(x));

%First Layer :: 784 Inputs
%Second Layer :: 30 Perceptrons Sigmoid Function 
%Third Layer :: 10 Perceptrons Softmax Function

%Hidden Layer Initialization
hiddenLayerPerceptrons=ones(30,1);

%Output Layer initialization
outputLayerPerceptrons=ones(10,1);


%Initializing weights for Input Layer
weightsInputToHiddenLayer=randn(inputsTrain,length(hiddenLayerPerceptrons),'double');
inputBias=1;
inputLayerBiasWeights=randn(length(hiddenLayerPerceptrons),1,'double');

%Initializing weights for Hidden Layer
weightsHiddenToOutputLayer=randn(length(hiddenLayerPerceptrons),length(outputLayerPerceptrons),'double');
hiddenLayerBias=1;
hiddenLayerBiasWeight=randn(length(outputLayerPerceptrons),1,'double');

%% Backpropagation Algorithm
epochs=5;

for k = 1:numOfTrainSamples    %Iterating all the training data
    for e = 1:epochs           %Maximum Epochs
     
        % The first coulmn of inputs is multiplied with the first coulmn of
        % weightsInputToHiddenLayer and then summed to get the input to first
        % perceptron of the hidden layer --->sumTemp
        %temp=ones(size(weightsInputToHiddenLayer));
        
        %Feed Forward
        for i=1:inputsTrain
            for j=1:length(hiddenLayerPerceptrons)
                tempInputs(i,j)=trainData(i,k).*weightsInputToHiddenLayer(i,j);
            end
        end
        
        sumTempInputs=sum(tempInputs);
        
        for i=1:length(hiddenLayerPerceptrons)
            tempBiasInput(i,1)=inputBias*inputLayerBiasWeights(i,1);
            hiddenLayerPerceptrons(i,1)=sig((sumTempInputs(1,i)+tempBiasInput(i,1)));
        end
        
        for i=1:length(hiddenLayerPerceptrons)
            for j=1:length(outputLayerPerceptrons)
                tempHidden(i,j)=hiddenLayerPerceptrons(i,1).*weightsHiddenToOutputLayer(i,j);
            end
        end
        
        sumTempHidden=transpose(sum(tempHidden));
        
        for i=1:length(outputLayerPerceptrons)
            tempBiasHidden(i,1)=hiddenLayerBias*hiddenLayerBiasWeight(i,1);
        end
        outputLayerPerceptrons=softmax((sumTempHidden+tempBiasHidden));
        
        %Total Error
        totalOutputError=0.5*(sum(outputLayerPerceptrons-vecTrainOutputs(:,1)));
     
        %Backward Pass
        %Calculating DeltaOutput and DeltaHidden 
        %Sk= Ok(1-Ok)(Ok-tk)
        deltaOutput=outputLayerPerceptrons.*(ones(size(outputLayerPerceptrons))-outputLayerPerceptrons).*(outputLayerPerceptrons-vecTrainOutputs(:,1));
             
        %Sj= Oj(1-Oj)*Sum(Sk*Wij) %Derivative of Error Function w.r.t Wij
        for i=1:length(hiddenLayerPerceptrons)    
            for j=1:length(outputLayerPerceptrons)
                tempsum(i,1)=deltaOutput(j,1)*weightsHiddenToOutputLayer(i,j);
            end
            temp2=sum(tempsum);
            deltaHidden(i,1)=hiddenLayerPerceptrons(i,1)*(1-hiddenLayerPerceptrons(i,1))*(temp2);
        end
        
        %Updating Input Weights based on Sj calculated above
        % Wij <--- Wij + eta(Sj * xij)
        for i=1:length(hiddenLayerPerceptrons)  % Weights of all inputs to 1st Hidden Perceptron are in 1st coulmn and so on 
            for j=1:inputsTrain
                weightsInputToHiddenLayer(j,i)= weightsInputToHiddenLayer(j,i) + (eta*(deltaHidden(i,1)*trainData(j,k)));
            end
        end
        
        %Updating Hidden Layer Weights based on Sk
        % Wjk <--- Wjk + eta(Sk * xjk)
        for i=1:length(outputLayerPerceptrons)  % Weights of all inputs to 1st Output Perceptron are in 1st coulmn and so on 
            for j=1:length(hiddenLayerPerceptrons)
                weightsHiddenToOutputLayer(j,i)= weightsHiddenToOutputLayer(j,i) + (eta*(deltaOutput(i,1)*hiddenLayerPerceptrons(j,1)));
            end
        end
        
        
    end
end


%% Testing
% 
% for k=1:numOfTestSamples  %for the whole test set
%     
%     for i=1:inputsTest  %for One input
%         for j=1:length(hiddenLayerPerceptrons)
%             tempTest(i,j)= testData(i,k).*weightsInputToHiddenLayer(i,j);
%         end
%     end
%     
%     
%         sumTempTest=sum(tempTest);
%         
%         for i=1:length(hiddenLayerPerceptrons)
%             %tempBiasInput(i,1)=inputBias*inputLayerBiasWeights(i,1);
%             testHiddenLayerPerceptrons(i,1)=sig((sumTempTest(1,i))); %+tempBiasInput(i,1)
%         end
%         
%         for i=1:length(hiddenLayerPerceptrons)
%             for j=1:length(outputLayerPerceptrons)
%                 tempHiddenTest(i,j)=testHiddenLayerPerceptrons(i,1).*weightsHiddenToOutputLayer(i,j);
%             end
%         end
%         
%         sumTempHiddenTest=transpose(sum(tempHiddenTest));
%         
%         for i=1:length(outputLayerPerceptrons)
%             tempBiasHidden(i,1)=hiddenLayerBias*hiddenLayerBiasWeight(i,1);
%         end
%         outputLayerPerceptrons=softmax((sumTempHiddenTest)); %+tempBiasHidden
%       
%             
% end


