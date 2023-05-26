$(document).ready(function() {
    
    $('#drs_select').change(formsSubmit);
    $('#vis_type').change(formsSubmit);

    function formsSubmit() {
        $("#drs-form").submit();
    }

});
