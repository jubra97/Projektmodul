% load subnets; comment out line for input checking; throws error but it
% works
global subnets layer_params

layer_params = {};
subnets = {};
% for i = 0:13
%     delete A
% end
for i = 0:13
    layer_params{i+1} = importONNXFunction(join(["onnx\", num2str(i), ".onnx"], ""), join(["ANARX_Layer", num2str(i)], ""));
    out = {}
    fid=fopen(join(["ANARX_Layer", num2str(i), ".m"], ""));
    line=fgetl(fid);
    k=1;
    comment_out_idx = 1;
    while ischar(line)
       out{k,1}=line;
       line=fgetl(fid);
       if strfind(line, "error(message('nnet_cnn_onnx:onnx:InputNeedsPermute',inputName, expectedSizeStr, inputSizeStr));")
          comment_out_idx = k + 1
       end
       k=k+1;
    end
    out{comment_out_idx} = join(["%", out{comment_out_idx}]);
    fclose(fid);
    fid= fopen(join(["ANARX_Layer", num2str(i), ".m"], ""),'w')
    for k=1:numel(out)
       fprintf(fid,'%s\n',out{k});
    end
    fclose(fid);
    subnets{i+1} = (join(["ANARX_Layer", num2str(i)], ""));
end

% Simulink.Bus.createObject(layer_params);

%%
a1 = extractdata(layer_params{1}.Learnables.x6(1))
a2 = extractdata(layer_params{1}.Learnables.x6(2))
b = extractdata(layer_params{1}.Nonlearnables.linear_layers_0_bias)

x = zeros(14, 1);
y = zeros(6000, 1);
inputs = zeros(6000, 1);
step = [zeros(2000, 1); 0.3.*ones(2000, 1); zeros(2000, 1)];
sinewave = sin(0:0.01:60);
inputs = u2_t;
% inputs = ones(6000, 1);
xs = zeros(14, 6000);

ramp_up = linspace(0, 1, 400)';
ramp_down = linspace(1, 0, 400)';
inputs = [zeros(400, 1); ramp_up; ones(400, 1); ramp_down; zeros(400, 1); ramp_up; ones(400, 1); ramp_down];

clear i
for j = 1:3200
   y(j) = x(1);
%    inputs(j) = (sinewave(j) - x(2) - x(1) * a2 - b) / a1;
   x(1) = 0;
   x = update_state(x, inputs(j));
   xs(:, j) = x;
end


function next_state = update_state(x, u)
    global subnets layer_params
    lag_map = [[1, 1]; [1, 1]; [1, 1]; [1, 1]; [1, 1]; [0, 1]; [0, 1]; [0, 1]; [0, 1]; [0, 1]; [0, 1]; [0, 1]; [0, 1]; [0, 1]];
    next_inputs = {};
    for i = 1:size(lag_map, 1)
        if lag_map(i, 1) == 1
            next_inputs{i} = [u, x(1)];
        else
            next_inputs{i} = [x(1)];
        end
    end
    next_state = [x(2:end, 1); 0];
    for i = 1:size(lag_map, 1)
        next_state(i) = next_state(i) + feval(subnets{i}, next_inputs{i}, layer_params{i});
    end


%     next_state = x;
%     for i = 1:size(lag_map, 1) - 1
%         next_state(i) = x(i+1) + feval(subnets{i}, next_inputs{i}, layer_params{i});
%     end
%     next_state(end) = feval(subnets{end}, next_inputs{end}, layer_params{end});
end