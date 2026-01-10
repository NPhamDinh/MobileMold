import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import 'camera_page.dart';
import 'info_gallery.dart';
import 'theme/constants.dart';

class MainScreen extends StatelessWidget {
  const MainScreen({
    super.key,
  });

  static const routeName = '/main';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      endDrawer: const SlideMenu(),
      appBar: AppBar(
        automaticallyImplyLeading: false,
        toolbarHeight: MediaQuery.of(context).size.height * 0.13,
        elevation: 5,
        shadowColor: Colors.black,
        backgroundColor: Colors.white,
        surfaceTintColor: Colors.white,
        title: Row(
          children: [
            Image(
              width: MediaQuery.of(context).size.width * 0.5,
              isAntiAlias: true,
              image: const AssetImage('assets/logos/logo_long.png'),
            )
          ],
        ),
      ),
      body: const SafeArea(child: CameraPage()),
    );
  }
}

class SlideMenu extends StatelessWidget {
  const SlideMenu({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Drawer(
      child: ListView(
        padding: EdgeInsets.zero,
        children: [
          const DrawerHeader(
            decoration: BoxDecoration(),
            child: Padding(
              padding: EdgeInsets.all(kDefaultPadding),
              child: Image(
                isAntiAlias: true,
                image: AssetImage('assets/logos/Logo V2.png'),
              ),
            ),
          ),
          ListTile(
            title: const Text('How does FreshLens work?'),
            onTap: () {
              // show InfoGallery
              Navigator.of(context).pushNamedAndRemoveUntil(
                  InfoGallery.routeName, (route) => false);
            },
          ),
          ListTile(
            title: const Text('Dataset license'),
            onTap: () {
              showDialog(
                context: context,
                builder: (context) => AlertDialog(
                    title: const Text("Dataset license"),
                    backgroundColor: Colors.white,
                    content: RichText(
                      text: TextSpan(
                        style: DefaultTextStyle.of(context).style,
                        children: [
                          const TextSpan(
                              text:
                                  'The dataset of this work is available on '),
                          WidgetSpan(
                            child: GestureDetector(
                              onTap: () => launchUrl(Uri.parse(
                                  'https://github.com/NPhamDinh/MobileMold')),
                              child: const Text(
                                'https://github.com/NPhamDinh/MobileMold',
                                style: TextStyle(
                                  color: Colors.blue,
                                  decoration: TextDecoration.underline,
                                ),
                              ),
                            ),
                          ),
                          const TextSpan(
                              text:
                                  ' and licensed under CC BY-NC-SA 4.0, visit '),
                          WidgetSpan(
                            child: GestureDetector(
                              onTap: () => launchUrl(Uri.parse(
                                  'https://creativecommons.org/licenses/by-nc-sa/4.0/')),
                              child: const Text(
                                'https://creativecommons.org/licenses/by-nc-sa/4.0/',
                                style: TextStyle(
                                  color: Colors.blue,
                                  decoration: TextDecoration.underline,
                                ),
                              ),
                            ),
                          ),
                          const TextSpan(text: '.'),
                        ],
                      ),
                    ),
                    actions: [
                      TextButton(
                          onPressed: Navigator.of(context).pop,
                          child: const Text("OK"))
                    ]),
              );
            },
          ),
          ListTile(
            title: const Text('About'),
            onTap: () {
              showAboutDialog(
                  applicationName: "FreshLens",
                  applicationVersion: "MVP",
                  applicationLegalese:
                      "This is a minimum viable product for demonstration purposes.",
                  context: context);
            },
          ),
        ],
      ),
    );
  }
}
