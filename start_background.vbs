' Starts featureclip completely hidden (no window). Used by the Windows autostart task.
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
folder = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = folder
sh.Run "cmd /c """ & folder & "\start_ui_hidden.bat""", 0, False
