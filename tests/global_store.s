
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_4>:
        s_load_b64 s[0:1], s[0:1], null                            // 000000001600: F4040000 F8000000
        v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 42              // 000000001608: CA100080 000000AA
        s_waitcnt lgkmcnt(0)                                       // 000000001610: BF89FC07
        global_store_b32 v0, v1, s[0:1]                            // 000000001614: DC6A0000 00000100
        s_nop 0                                                    // 00000000161C: BF800000
        s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001620: BFB60003
        s_endpgm                                                   // 000000001624: BFB00000

