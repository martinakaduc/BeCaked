$(document).ready(function(){
    $("#file-form-1").submit(function(e){
        e.preventDefault();
        var formData = new FormData($(this)[0]);
        $.ajax({
            url: '/upload-form-1',
            type: 'POST',
            data: formData,
            async: false,
            cache: false,
            contentType: false,
            enctype: 'multipart/form-data',
            processData: false,
            success: function (response) {
                remove_upload_1();
                $("#msg-form-1 > p").html('<i class="fa fas fa-check-circle" style = "color: green; font-size: 60px;"></i><br><p style = "color: green; margin-top: 20px;">Xử lý thành công</p>')
            },
            error: function (e){
                remove_upload_1();
                $("#msg-form-1 > p").html('<i class="fa fas fa-exclamation-circle" style = "color: red; font-size: 60px;"></i><br><p style = "color: red; margin-top: 20px;">Tệp bị lỗi</p>')
            }
        });
    })
    $("#file-form-2").submit(function(e){
        e.preventDefault();
        var formData = new FormData($(this)[0]);
        $.ajax({
            url: '/upload-form-2',
            type: 'POST',
            data: formData,
            async: false,
            cache: false,
            contentType: false,
            enctype: 'multipart/form-data',
            processData: false,
            success: function (response) {
                remove_upload_2();
                $("#msg-form-2 > p").html('<i class="fa fas fa-check-circle" style = "color: green; font-size: 60px;"></i><br><p style = "color: green; margin-top: 20px;">Xử lý thành công</p>')
            },
            error: function (e){
                remove_upload_2();
                $("#msg-form-2 > p").html('<i class="fa fas fa-exclamation-circle" style = "color: red; font-size: 60px;"></i><br><p style = "color: red; margin-top: 20px;">Tệp bị lỗi</p>')
            }
        });
    })
    $("#file-form-3").submit(function(e){
        e.preventDefault();
        var formData = new FormData($(this)[0]);
        $.ajax({
            url: '/upload-form-3',
            type: 'POST',
            data: formData,
            async: false,
            cache: false,
            contentType: false,
            enctype: 'multipart/form-data',
            processData: false,
            success: function (response) {
                remove_upload_3();
                $("#msg-form-3 > p").html('<i class="fa fas fa-check-circle" style = "color: green; font-size: 60px;"></i><br><p style = "color: green; margin-top: 20px;">Xử lý thành công</p>')
            },
            error: function (e){
                remove_upload_3();
                $("#msg-form-3 > p").html('<i class="fa fas fa-exclamation-circle" style = "color: red; font-size: 60px;"></i><br><p style = "color: red; margin-top: 20px;">Tệp bị lỗi</p>')
            }
        });
    })
})