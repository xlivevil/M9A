# MirrorChyan Usage Guide

- [MirrorChyan Usage Guide](#mirrorchyan-usage-guide)
  - [Using on MaaFramework Generic UI](#using-on-maaframework-generic-ui)
  - [Using the MirrorChyan Website](#using-the-mirrorchyan-website)

MirrorChyan is a third-party application distribution platform that simplifies updates for open-source applications.
Users pay to use it, and the revenue is shared with developers. Additionally, MirrorChyan itself is open-source.

By accessing or using the service, you agree to be bound by all terms of [this agreement](https://mirrorchyan.com/disclaimer.html).

## Using on MaaFramework Generic UI

Some MaaFramework generic UIs, such as [MFAAvalonia](https://github.com/SweetSmellFox/MFAAvalonia), [MFW-PyQt6](https://github.com/overflow65537/MFW-PyQt6), and [MFAWPF](https://github.com/SweetSmellFox/MFAWPF), have the functionality to update the UI and resources (like M9A) through MirrorChyan. You can configure this in the respective settings interface.

## Using the MirrorChyan Website

[Official Website](https://mirrorchyan.com). Most functions are displayed on the homepage. Here's an introduction to each:

1. Purchase Monthly Pass
    You can purchase directly from the homepage.
2. Query CDK
    [Query CDK](https://mirrorchyan.com/en/get-key). After purchase, you need to query the CDK using the order number (**Note: this and the following refer to the merchant order number**). This code is required for use.
    [Query Order Number](https://afdian.com/dashboard/order). If you don't know the order number, you can find it here or check the order details in your payment channel (e.g., Alipay).
3. Download the full M9A package
    [Download link](https://mirrorchyan.com/en/download?rid=M9A)  
    The default download of Windows x64 version, and other systems and architectures can pass parameters such as `os` (mac/linux) and `arch` (win64/arm64).

    ```bash
    # Example link
    https://mirrorchyan.com/en/download?rid=M9A&os=mac&arch=arm64
    ```

4. Order Transfer (Redemption code exchange is also here)
    [Order Transfer](https://mirrorchyan.com/en/transfer). Enter the old order number/redemption code on the left, and the new order number to transfer to on the right. This transfers the remaining validity period to the new order number.
5. Communication and Feedback
    [MirrorChyan Usage Communication and Feedback](https://pd.qq.com/g/MirrorChyan) (QQ Channel)
