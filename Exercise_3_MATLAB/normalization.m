function normalized = normalization( O,minOut,maxOut )
%NORMALIZATION 
%input:
%      O:   the matrix to be normalize, original matrix
% minOut:   the minimum value after normalized
% maxOut:   the maximum value after normalized
%output:   
%normalized:the normalized matrix

minO=min(min(O));maxO=max(max(O));
normalized=(O-minO)./(maxO-minO)*(maxOut-minOut)+minOut;
end

