#pragma once

#include "httplib.h"

struct SDSvrParams;

class SDHttpServer final : public httplib::Server {
public:
    explicit SDHttpServer(const SDSvrParams& params);

private:
    bool process_and_close_socket(socket_t socket) override;

    static thread_local bool close_after_response_;
};
