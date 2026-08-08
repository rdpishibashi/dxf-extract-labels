#define MyAppName "DXF Extract Labels"
#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif
#define MyAppPublisher "ULVAC"
#define MyAppExeName "DXF-Extract-Labels.exe"

[Setup]
AppId={{30D6D124-86DB-4C5D-A31D-0DC547043B3D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\DXF-Extract-Labels
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
OutputDir=Output
OutputBaseFilename=DXF-Extract-Labels-Setup-{#MyAppVersion}
UninstallDisplayIcon={app}\{#MyAppExeName}
SetupLogging=yes

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"

[Tasks]
Name: "desktopicon"; Description: "デスクトップにショートカットを作成"; GroupDescription: "追加タスク:"; Flags: unchecked

[InstallDelete]
Type: filesandordirs; Name: "{app}\\_internal"
Type: files; Name: "{app}\\{#MyAppExeName}"
Type: files; Name: "{app}\\app_icon.ico"

[Files]
Source: "..\dist\DXF-Extract-Labels.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\assets\app_icon.ico"; DestDir: "{app}"; Flags: ignoreversion
[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\app_icon.ico"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\app_icon.ico"; Tasks: desktopicon

[Code]
procedure CleanupPyInstallerTemp;
var
  FindRec: TFindRec;
  TempRoot: string;
  Candidate: string;
begin
  TempRoot := ExpandConstant('{localappdata}') + '\Temp';
  if FindFirst(TempRoot + '\_MEI*', FindRec) then
  begin
    try
      repeat
        if (FindRec.Name <> '.') and (FindRec.Name <> '..') then
        begin
          Candidate := TempRoot + '\' + FindRec.Name;
          if FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY <> 0 then
            DelTree(Candidate, True, True, True)
          else
            DeleteFile(Candidate);
        end;
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssInstall then
  begin
    Exec('taskkill.exe', '/F /IM DXF-Extract-Labels.exe', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    CleanupPyInstallerTemp;
  end;
end;

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{#MyAppName} を起動"; Flags: nowait postinstall skipifsilent
