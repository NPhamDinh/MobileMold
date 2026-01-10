import 'package:flutter/material.dart';
import 'package:freshlens/src/ui/preview_page.dart';
import 'package:freshlens/src/ui/theme/light_theme.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'src/ui/info_gallery.dart';
import 'src/ui/main_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final SharedPreferences prefs = await SharedPreferences.getInstance();
  bool? hasSeenInfoGallery = prefs.getBool('hasSeenInfoGallery');
  runApp(MainApp(
      startInfoGallery: hasSeenInfoGallery == null || !hasSeenInfoGallery));
}

class MainApp extends StatelessWidget {
  final bool startInfoGallery;
  const MainApp({super.key, required this.startInfoGallery});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: lightTheme,
      darkTheme: lightTheme,
      themeMode: ThemeMode.light,
      onGenerateRoute: (settings) {
        debugPrint(settings.toString());
        var routeName = settings.name;
        if (routeName == '/') {
          routeName =
              startInfoGallery ? InfoGallery.routeName : MainScreen.routeName;
        }

        if (routeName == PreviewPage.routeName) {
          final imagePath = settings.arguments as String;
          return MaterialPageRoute(
            builder: (context) {
              return PreviewPage(
                imagePath: imagePath,
              );
            },
          );
        }

        if (routeName == InfoGallery.routeName) {
          return MaterialPageRoute(
            builder: (context) {
              return const InfoGallery();
            },
          );
        }

        if (routeName == MainScreen.routeName) {
          return MaterialPageRoute(
            builder: (context) {
              return const MainScreen();
            },
          );
        }

        assert(false, 'Need to implement ${settings.name}');
        return null;
      },
      initialRoute: "/",
    );
  }
}
