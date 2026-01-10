import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:freshlens/src/ui/theme/constants.dart';

class BottomPanel extends StatefulWidget {
  final Future<String?> Function() onTakePhotoPressed;
  final Future<String?> Function() onImportPhotoPressed;
  final void Function(CameraDescription) selectNewCamera;
  final List<CameraDescription> cameras;
  const BottomPanel({
    super.key,
    required this.onTakePhotoPressed,
    required this.onImportPhotoPressed,
    required this.cameras,
    required this.selectNewCamera,
  });

  @override
  State<BottomPanel> createState() => _BottomPanelState();
}

class _BottomPanelState extends State<BottomPanel> {
  bool _isLoadingImport = false;
  bool _isLoadingTake = false;
  int _selectedCameraIndex = 0;

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;
    return Expanded(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: kDefaultPadding),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            Padding(
              padding: const EdgeInsets.all(kDefaultPadding / 2),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  OutlinedButton(
                    onPressed: () async {
                      setState(() => _isLoadingTake = true);
                      String? imagePath = await widget.onImportPhotoPressed();
                      if (mounted) {
                        setState(() => _isLoadingTake = false);
                        if (imagePath == null) return;
                        Navigator.of(context)
                            .pushNamed('/result', arguments: imagePath);
                      }
                    },
                    child: Container(
                      alignment: Alignment.center,
                      constraints: BoxConstraints(maxWidth: size.width * 0.15),
                      child: _isLoadingTake
                          ? const SizedBox.square(
                              dimension: 20, child: CircularProgressIndicator())
                          : const Text(
                              "IMPORT PICTURE",
                              textAlign: TextAlign.center,
                            ),
                    ),
                  ),
                  MaterialButton(
                    onPressed: () async {
                      setState(() => _isLoadingImport = true);
                      String? imagePath = await widget.onTakePhotoPressed();
                      if (mounted) {
                        setState(() => _isLoadingImport = false);
                        if (imagePath == null) return;
                        Navigator.of(context)
                            .pushNamed('/result', arguments: imagePath);
                      }
                    },
                    elevation: kElevation,
                    color: Theme.of(context).colorScheme.primary,
                    textColor: Colors.white,
                    padding: const EdgeInsets.all(kDefaultPadding),
                    shape: const CircleBorder(),
                    child: _isLoadingImport
                        ? const CircularProgressIndicator()
                        : Icon(
                            Icons.camera_alt_outlined,
                            size: size.width * 0.14,
                          ),
                  ),
                  OutlinedButton(
                    onPressed: () {
                      showDialog(
                          context: context,
                          builder: (context) {
                            // convert camera descriptions to a list of strings
                            List<String> cameraNames = [
                              for (final desc in widget.cameras)
                                desc.lensDirection == CameraLensDirection.front
                                    ? "Front (${desc.name})"
                                    : "Back (${desc.name})"
                            ];
                            // show a Dialog with a dropdown menu to select the camera
                            return Dialog(
                              surfaceTintColor:
                                  Theme.of(context).colorScheme.surface,
                              backgroundColor:
                                  Theme.of(context).colorScheme.surface,
                              child: Container(
                                padding: const EdgeInsets.all(kDefaultPadding),
                                child: Column(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Text(
                                      "SELECT CAMERA",
                                      style: Theme.of(context)
                                          .textTheme
                                          .headlineMedium,
                                    ),
                                    const SizedBox(
                                      height: 2 * kDefaultPadding,
                                    ),
                                    SizedBox(
                                        width: double.infinity,
                                        child: DropdownButtonFormField(
                                          borderRadius: const BorderRadius.all(
                                              Radius.circular(kBorderRadius)),
                                          isExpanded: true,
                                          itemHeight:
                                              kMinInteractiveDimension * 2,
                                          iconEnabledColor: Theme.of(context)
                                              .colorScheme
                                              .primary,
                                          value: _selectedCameraIndex,
                                          items: [
                                            for (int i = 0;
                                                i < cameraNames.length;
                                                i++)
                                              DropdownMenuItem(
                                                value: i,
                                                child: Text(cameraNames[i]),
                                              ),
                                          ],
                                          onSaved: (value) {
                                            setState(() =>
                                                _selectedCameraIndex =
                                                    value as int);
                                          },
                                          onChanged: (value) {
                                            setState(() =>
                                                _selectedCameraIndex =
                                                    value as int);
                                          },
                                        )),
                                    const SizedBox(
                                      height: 2 * kDefaultPadding,
                                    ),
                                    OutlinedButton(
                                      onPressed: () {
                                        if (widget.cameras.isNotEmpty) {
                                          widget.selectNewCamera(widget
                                              .cameras[_selectedCameraIndex]);
                                        }
                                        Navigator.of(context).pop();
                                      },
                                      child: const Text("OK"),
                                    ),
                                  ],
                                ),
                              ),
                            );
                          });
                    },
                    child: Container(
                      alignment: Alignment.center,
                      width: size.width * 0.15,
                      child: const Text(
                        "SWITCH CAMERA",
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            Text(
              "Press the camera button to detect mold. Make sure the microscope is close enough to the object to focus.",
              style: Theme.of(context)
                  .textTheme
                  .bodyMedium
                  ?.copyWith(fontSize: size.height * 0.02),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
