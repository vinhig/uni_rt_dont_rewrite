#include <stdint.h>
#include <stdio.h>

#include <stdlib.h>

int main(int argc, char const *argv[]) {
#define ORSS_MIGRATION_TABLE_SIZE 1024
  struct PacketBuffer {
    struct Parsed_Packet *packets[1024];
    uint64_t write_head;
  };

  struct PacketBuffer* buffer_table = (struct PacketBuffer*)malloc(sizeof(struct PacketBuffer) * 1024);

  printf("%lu\n", sizeof(struct PacketBuffer) * 1024 / 1024 / 1024);
  return 0;
}
