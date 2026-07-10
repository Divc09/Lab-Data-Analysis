Option Explicit

Dim fileSystem, shell, projectRoot, parentRoot
Dim localPython, baselinePython, pythonExe, command, launchResult

Set fileSystem = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

projectRoot = fileSystem.GetParentFolderName(WScript.ScriptFullName)
parentRoot = fileSystem.GetParentFolderName(projectRoot)
localPython = fileSystem.BuildPath(projectRoot, ".venv\Scripts\pythonw.exe")
baselinePython = fileSystem.BuildPath(parentRoot, "Lab_Data_Analysis\.venv\Scripts\pythonw.exe")

If fileSystem.FileExists(localPython) Then
    pythonExe = localPython
ElseIf fileSystem.FileExists(baselinePython) Then
    pythonExe = baselinePython
Else
    MsgBox "SmartFitter could not find its Python environment." & vbCrLf & vbCrLf & _
        "Expected one of:" & vbCrLf & _
        localPython & vbCrLf & _
        baselinePython & vbCrLf & vbCrLf & _
        "Create the .venv environment and install requirements.txt, then try again.", _
        vbExclamation, "SmartFitter launcher"
    WScript.Quit 1
End If

If WScript.Arguments.Named.Exists("check") Then
    WScript.Echo "Project: " & projectRoot & vbCrLf & "Python: " & pythonExe
    WScript.Quit 0
End If

shell.CurrentDirectory = projectRoot
command = Chr(34) & pythonExe & Chr(34) & " -m nvfit.gui_app"

On Error Resume Next
launchResult = shell.Run(command, 0, False)
If Err.Number <> 0 Then
    MsgBox "SmartFitter could not be started." & vbCrLf & vbCrLf & Err.Description, _
        vbCritical, "SmartFitter launcher"
    WScript.Quit 1
End If
On Error GoTo 0
