
SAMPLE_TIME = 0.0002;

INC_TURN = 2^20;
UMFANG = 0.1 * pi;
FAKTOR_m = UMFANG / INC_TURN; % Inc * FAKTOR_m = m 

mFlange = 0.018;            % 18g   (gemessen)


%  ______________________________________________________________________
%% -------------------------- AUFBAU  (CONFIG) --------------------------
%  ______________________________________________________________________


% Manuell bestimmte Inkrementalgeberposition bei Seill�nge Null (Seil ist
% am Austrittspunkt)
zero_poses = [...
    3143153400]; % Increments

MOTOREN_ANZAHL = numel(zero_poses);

%  _______________________________________________________________________
%% ------------------------------ Parameter ------------------------------
%  _______________________________________________________________________
MAXIMUM_TORQUE = 950;


%  _______________________________________________________________________
%% ------------------------- Sicherheitssysteme -------------------------
%  _______________________________________________________________________
% Geschwindigkeitsbegrenzung pro Seil -> Bremse beim �berschreiten
WS_MAX_VEL_m_s = 0.4;  % Meter/s

WS_MIN_LENGTHS = [-0.2];
WS_MAX_LENGTHS = [-0.1];



%  ______________________________________________________________________
%% ---------------------------- LOAD Config -----------------------------
%  ______________________________________________________________________
%% Generiere zufällige Reihe von Vorgabewerten

rng(123654); % Zufallsgenerator für reproduzierbare Ergebnisse initialisieren

upper = 1; % Oberes Limit
lower = 0.1; % Unteres Limit
n = 10000; % Anzahl der Werte

R = (upper-lower).*rand(1, n) + lower;

%% Parameter
duration = 1; % Zeit zwischen Steps
T = 0.1; % Zeitkonstante der PT2-Interpolation
