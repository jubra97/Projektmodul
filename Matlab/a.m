load("new_sesnor_steps_with_actual_torque_long_pause.mat")
data = data_3;

figure
data(1, :) = data(1, :) - data(1, 1);
plot(data(1, :), data(4, :));
hold on
plot(data(1, :), data(2, :) * 1e3);


sig_start = 485.3265 * 4000;
sig_stop = sig_start + 10 * 4000;

step_data = data(:, sig_start:sig_stop);

figure
plot(step_data(1, :), step_data(4, :))
hold on
plot(step_data(1, :), step_data(2, :) * 1e4);

t = step_data(1, :) - step_data(1, 1);
u = step_data(2, :) - step_data(2, 1);
y = step_data(4, :) - mean(step_data(4, 1:1000));

figure
plot(u)
hold on
plot(y)

to_simulink = [t; u; y]';