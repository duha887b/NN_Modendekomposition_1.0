% Training of dlnet-type neural network for mode decomposition
%@Dustin Hanusch
clear all
close all
%% load dataset

load("networks.mat");
%  2. define the input and output size for neural network
Nmodes = 5; % 3 oder 5
moreData = 0; % 0= 10000 , 1=50000  Trainingsdaten

if Nmodes == 3 
    load("mmf_Traingsdata_3modes.mat");     %  1. load the dataset
end
if Nmodes == 5
    if moreData == 0
        load("mmf_Traingsdata_5modes.mat");
    end
    if moreData == 1
        load("mmf_Traingsdata_5modes_50000.mat");
       
    end
end

netarc = 1; %1:VGG 0:MLP

ImageSize = 32;
outputsize = Nmodes*2-1;
inputsize = ImageSize.^2;
inputSize = [32 32 1];
%% create MLP neural network - step 3 

Layers_MLP = [

    imageInputLayer(inputSize,Normalization="none" )

    fullyConnectedLayer(inputsize,'Name','fc1')
    leakyReluLayer('Name', 'relu1')
    fullyConnectedLayer(inputsize,'Name','fc2')
    leakyReluLayer('Name','relu2')
    fullyConnectedLayer(outputsize,"Name","fc_output")

    sigmoidLayer("Name",'out')
   
];

%% create VGG neural network - step 7
Layers_VGG= [
    
    imageInputLayer(inputSize,Normalization="none" )

    convolution2dLayer(3,64, 'Name',"conv1_1","Padding","same")
    reluLayer('Name',"relu1_1")
    convolution2dLayer(3,64,'Name',"conv1_2","Padding","same")
    reluLayer('Name',"relu1_2")
    maxPooling2dLayer(2,"Stride",2,'Name',"pooling1")

    convolution2dLayer(3,128, 'Name',"conv2_1","Padding","same")
    reluLayer('Name',"relu2_1")
    convolution2dLayer(3,128,'Name',"conv2_2","Padding","same")
    reluLayer('Name',"relu2_2")
    maxPooling2dLayer(2,"Stride",2,'Name',"pooling2")

    convolution2dLayer(3,256, 'Name',"conv3_1","Padding","same")
    reluLayer('Name',"relu3_1")
    convolution2dLayer(3,256,'Name',"conv3_2","Padding","same")
    reluLayer('Name',"relu3_2")
    maxPooling2dLayer(2,"Stride",2,'Name',"pooling3")

    fullyConnectedLayer(256,'Name',"fc1")
    dropoutLayer(0.5,'Name',"drop1")
    fullyConnectedLayer(128,'Name',"fc2")
    dropoutLayer(0.5,'Name',"drop2")
    fullyConnectedLayer(outputsize,'Name',"fc_output")

    sigmoidLayer("Name",'out')
];

%analyzeNetwork(Layers_VGG);
%% use command dlnetwork Layers_VGG oder Layers_MLP ()

if netarc == 0
    lgraph = layerGraph(Layers_MLP);
end
if netarc == 1
    lgraph = layerGraph(Layers_VGG);
end
dlnet = dlnetwork(lgraph);

%% learnable parameters transfer  - step 8 & 9
% use Transfer Learning
old_dlnet = vgg_5modes_d_TL1;             %von welchem netz soll abgeleitet werden 
old_LayerArray = old_dlnet.Layers;
LayerArray = dlnet.Layers;
valueoldex = table2array(old_dlnet.Learnables(:,1));
valcount = 2;
for iL=2: size(LayerArray,1)-1
    oldProp =  old_LayerArray(iL);
    newProp = LayerArray(iL);
    if newProp.Name == oldProp.Name
        
        if newProp.Name == valueoldex(valcount)
            oldLearn = cell2mat(old_dlnet.Learnables.Value(valcount));
            newLearn = cell2mat(dlnet.Learnables.Value(valcount));
            
            if size(newLearn)~=0
                if size(oldLearn) == size(newLearn)
                    dlnet.Learnables.Value(valcount) = old_dlnet.Learnables.Value(valcount);
                    dlnet.Learnables.Value(valcount-1) = old_dlnet.Learnables.Value(valcount-1);
                    disp(valcount);
                end
            end
            valcount = valcount +2;
        end
    end
end


%% Training network  - step 3
% define hyperparameters

miniBatchSize = 128;

numEpochs = 5;

learnRate = 0.001;

numObservations = size(XTrain,4);

numIterationsPerEpoch = floor(numObservations./miniBatchSize);

executionEnvironment = "parallel";

ValidationFrequenz = 50;
ValCount = 0;
dlValid = dlarray(XValid,'SSCB');


%Visualize the training progress in a plot.
plots = "training-progress";
% Train Network
if plots == "training-progress"
    figure
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098],'LineWidth',1.5);
    lineLossValid = animatedline('Color',[0 0 0],'LineWidth',1.5,'LineStyle','-.');
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    legend('TrainingLoss','ValidationLoss')
    grid on
end
iteration = 0;
start = tic;
% Train Network
% Initialize the average gradients and squared average gradients.
averageGrad = [];
averageSqGrad = [];
for epoch = 1:numEpochs
    disp(epoch);
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        
        % 1. Read mini-batch of data and convert the labels to dummy
        % variables.
        
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        XTmp = XTrain(:,:,:,idx);
        
        
        Y = zeros(miniBatchSize,outputsize,"double");
        Y = YTrain(idx,:);
        Y = Y';

        % 2. Convert mini-batch of data to a dlarray.
        dlX = dlarray(XTmp,'SSCB');
        % If training on a GPU, then convert data to a gpuArray.

        % 3. Evaluate the model gradients and loss using the
        % modelGradients() and dlfeval()
        [gradients,loss] = dlfeval(@modelGradients,dlnet,dlX,Y);
        % 4. Update the network parameters using the Adam optimizer.
        [dlnet,averageGrad,averageSqGrad ] = adamupdate(dlnet,gradients,averageGrad,averageSqGrad,iteration,learnRate);
        
        %Validation
        if iteration >= ValCount
            dlYValid = predict(dlnet,dlValid);
            ValLoss = mse(dlYValid,YValid');
            ValCount = ValCount + ValidationFrequenz;
        end
        
        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
            addpoints(lineLossValid,iteration,double(gather(extractdata(ValLoss))))
            title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", TrainingLoss: " + num2str(double(gather(extractdata(loss)))) + ", ValidationLoss: " + num2str(double(gather(extractdata(ValLoss)))));
            drawnow
        end
    end
end


%% Test Network  - step 4
dlnet = vgg_5modes_d_TL2;
% transfer data to dlarray
dlTest = dlarray(XTest,'SSCB');
% use command "predict"
dlYPred = predict(dlnet,dlTest);
% use command "extractdata" to extract data from dlarray
YPred = zeros(2*Nmodes-1,size(YTest,1),"double");
YPred = double(extractdata(dlYPred))';
% reconstruct field distribution

[Image_data_complex,complex_vector_N] = mmf_rebuilt_image(YPred,XTest,Nmodes);

figure
k=0;
for imt=1:5
    k=k+1;
    subplot(5,2,k), imshow(Image_data_complex(:,:,1,imt),[0 1]),title('Pred')
    k=k+1;
    subplot(5,2,k), imshow(XTest(:,:,1,imt),[0 1]),title('Org')
end
%%  Visualization results - step 5


for itc=1:size(XTest,4)
    corr_gt_rc(itc) = corr2(Image_data_complex(:,:,1,itc),XTest(:,:,1,itc));    % calculate Correlation between the ground truth and reconstruction                         
end

%std der ergebnisse
std_corr = std(corr_gt_rc);
mean_corr = mean(corr_gt_rc);

% calulate relative error of ampplitude and phase
phase_rel = abs(mean(YTest(:,Nmodes+1:end)) - mean(YPred(:,Nmodes+1:end)))./abs(mean(YTest(:,Nmodes+1:end)));
ampli_rel = abs(mean((YTest(:,1:Nmodes))) - mean((YPred(:,1:Nmodes))))./abs(mean(YTest(:,1:Nmodes)));

mean_phase = mean(phase_rel);
mean_ampli = mean(ampli_rel);

%display values
disp("Mean_corr: " + num2str(mean_corr) + "  Std: " + num2str(std_corr) + ...
    "  Mean_amp: " + num2str(mean_ampli) + "  Mean_phase: " + num2str(mean_phase))


% plot()
figure
subplot(1,1,1), boxchart(corr_gt_rc),title('Correlation') ;
%subplot(1,2,2), boxchart(std_t_t),title('Standartabweichung');



%% save model

%vgg_5modes = dlnet;
%vgg_3modes = dlnet;
%mlp_3modes = dlnet;
%mlp_5modes = dlnet;
%vgg_5modes_TL1 = dlnet;
%vgg_5modes_TL2 = dlnet;rrrrrrr
%vgg_3modes_d = dlnet;
%vgg_5modes_d = dlnet;
%vgg_5modes_d_TL1 = dlnet;

%vgg_5modes_d_TL2 = dlnet;


save("networks.mat",'mlp_3modes','mlp_5modes','vgg_3modes','vgg_5modes','vgg_5modes_TL1','vgg_5modes_TL2', ...
    'vgg_3modes_d','vgg_5modes_d','vgg_5modes_d_TL1','vgg_5modes_d_TL2');

