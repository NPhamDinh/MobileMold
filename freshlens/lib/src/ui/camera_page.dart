import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:freshlens/src/ui/bottom_panel.dart';
import 'package:freshlens/src/ui/theme/constants.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});

  @override
  CameraPageState createState() => CameraPageState();
}

class CameraPageState extends State<CameraPage> with WidgetsBindingObserver {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  static double zoomFactor = 3;

  @override
  void initState() {
    super.initState();
    debugPrint("initState cameraPage");
    _checkPermissionsAndInitialize().then((_) {
      // didChangeAppLifecycleState will be called only after adding observer
      // sadly, still after permission grant dialog closed
      WidgetsBinding.instance.addObserver(this);
    }); // Check permissions before initializing camera
  }

  Future<void> _checkPermissionsAndInitialize() async {
    // Request camera permission
    debugPrint("Requesting camera permission");
    await Permission.camera.request();
    debugPrint("Camera permission requested");

    // Check permission status
    if (await Permission.camera.isGranted) {
      await initCamera();
    }
  }

  Future<void> initCamera() async {
    _cameras = await availableCameras();
    if (_cameras!.isEmpty) {
      // IOS simulator has no camera ...
      return;
    }
    // Initialize the camera with the first camera in the list
    var backCamera = _cameras!.firstWhere(
        (element) => element.lensDirection == CameraLensDirection.back,
        orElse: () => _cameras!.first);
    await onNewCameraSelected(backCamera);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // dont return without initializing the camera!
    // This caused the grey screen when resuming, since we returned without reinitializing the camera
    //if (_controller == null || !_controller!.value.isInitialized) return;
    debugPrint("didChangeAppLifecycleState $state");
    if ([
      AppLifecycleState.inactive,
      AppLifecycleState.paused,
      AppLifecycleState.hidden,
      AppLifecycleState.detached
    ].contains(state)) {
      // Free up memory when camera not active
      _controller?.dispose();
      _controller = null;
    } else if (state == AppLifecycleState.resumed) {
      // Reinitialize the camera with same properties
      debugPrint("resumed, $_controller");
      initCamera();
      //onNewCameraSelected(_controller!.description);
    }
  }

  @override
  void dispose() {
    debugPrint("disposed camera page $mounted");
    _controller?.dispose();
    _controller = null;
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  Future<XFile?> capturePhoto() async {
    if (_controller == null) return null;
    final CameraController? cameraController = _controller;
    if (cameraController!.value.isTakingPicture) {
      // A capture is already pending, do nothing.
      return null;
    }
    try {
      await cameraController.setFlashMode(FlashMode.off);
      XFile file = await cameraController.takePicture();
      return file;
    } on CameraException catch (e) {
      debugPrint('Error occured while taking picture: $e');
      return null;
    }
  }

  Future<String?> _onTakePhotoPressed() async {
    final xFile = await capturePhoto();
    debugPrint("xFile: ${xFile?.path}");
    if (xFile != null && xFile.path.isNotEmpty) return xFile.path;
    return null;
  }

  Future<String?> _onImportPhotoPressed() async {
    late final XFile? image;
    try {
      final ImagePicker picker = ImagePicker();
      image = await picker.pickImage(source: ImageSource.gallery);
    } on PlatformException {
      if (mounted) {
        showDialog(
            context: context,
            builder: (context) {
              return AlertDialog(
                title: const Text("Error"),
                content: const Text(
                    "Could not open gallery. Please check your permissions in the settings app."),
                actions: [
                  TextButton(
                      onPressed: () {
                        Navigator.of(context).pop();
                      },
                      child: const Text("OK"))
                ],
              );
            });
      }
      return null;
    }

    if (image != null && image.path.isNotEmpty) return image.path;
    return null;
  }

  @override
  Widget build(BuildContext context) {
    var size = MediaQuery.of(context).size;
    return Column(children: [
      Container(
        color: Colors.grey,
        height: size.width,
        width: double.infinity,
        child: SizedBox.square(
          dimension: size.width,
          child: ClipRect(
            child: OverflowBox(
              alignment: Alignment.center,
              child: FittedBox(
                fit: BoxFit.fitWidth,
                child: _controller == null || !_controller!.value.isInitialized
                    ? Container(
                        width: size.width,
                        padding: const EdgeInsets.all(kDefaultPadding),
                        alignment: Alignment.center,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.center,
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const CircularProgressIndicator(),
                            const SizedBox(
                              height: kDefaultPadding,
                            ),
                            Text(
                              "A camera preview should show up here. FreshLens needs to access your camera to take a picture for mold detection. If you have not granted camera access, change it in the settings app.",
                              maxLines: 5,
                              softWrap: true,
                              style: Theme.of(context)
                                  .textTheme
                                  .bodyMedium
                                  ?.copyWith(fontWeight: FontWeight.bold),
                            ),
                          ],
                        ),
                      )
                    : SizedBox(
                        width: size.width / _controller!.value.aspectRatio,
                        height: size.width,
                        child: CameraPreview(_controller!),
                      ),
              ),
            ),
          ),
        ),
      ),
      BottomPanel(
        onTakePhotoPressed: _onTakePhotoPressed,
        onImportPhotoPressed: _onImportPhotoPressed,
        cameras: _cameras ?? [],
        selectNewCamera: onNewCameraSelected,
      ),
    ]);
  }

  Future<void> onNewCameraSelected(CameraDescription description) async {
    if (_controller != null) debugPrint("disposing old controller");
    await _controller?.dispose();
    _controller = null;

    // Instantiating the camera controller
    debugPrint(description.name);
    _controller = CameraController(
      description,
      ResolutionPreset.high,
      imageFormatGroup:
          Platform.isIOS ? ImageFormatGroup.bgra8888 : ImageFormatGroup.jpeg,
      enableAudio: false,
    );

    // Initialize controller
    try {
      await _controller!.initialize();
      _controller!.setZoomLevel(zoomFactor);
      _controller!.lockCaptureOrientation(DeviceOrientation.portraitUp);
    } on CameraException catch (e) {
      debugPrint('Error initializing camera: $e');
      _controller = null;
      return;
    }

    // cameraController.setZoomLevel(10);
    if (mounted) setState(() {});

    // Update UI if controller updated
    _controller!.addListener(() {
      if (mounted) setState(() {});
    });
  }
}
