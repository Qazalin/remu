
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_256_256n1>:
	v_mov_b32_e32 v0, 0xff800000                               // 000000001600: 7E0002FF FF800000
	s_mov_b32 s2, s15                                          // 000000001608: BE82000F
	s_movk_i32 s3, 0x100                                       // 00000000160C: B0030100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001610: BF8704A1
	v_max_f32_e32 v0, v0, v0                                   // 000000001614: 20000100
	s_sub_i32 s3, s3, 32                                       // 000000001618: 8183A003
	s_cmp_eq_u32 s3, 0                                         // 00000000161C: BF068003
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001620: BF870001
	v_max_f32_e32 v0, 1.0, v0                                  // 000000001624: 200000F2
	s_cbranch_scc0 65529                                       // 000000001628: BFA1FFF9 <r_256_256n1+0x10>
	s_load_b64 s[0:1], s[0:1], null                            // 00000000162C: F4040000 F8000000
	s_ashr_i32 s3, s2, 31                                      // 000000001634: 86039F02
	v_mov_b32_e32 v1, 0                                        // 000000001638: 7E020280
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000163C: 84828202
	s_waitcnt lgkmcnt(0)                                       // 000000001640: BF89FC07
	s_add_u32 s0, s0, s2                                       // 000000001644: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001648: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 00000000164C: DC6A0000 00000001
	s_nop 0                                                    // 000000001654: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001658: BFB60003
	s_endpgm                                                   // 00000000165C: BFB00000
