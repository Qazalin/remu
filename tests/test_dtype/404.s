
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n52>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	v_mov_b32_e32 v0, 0                                        // 000000001610: 7E000280
	s_lshl_b64 s[6:7], s[4:5], 1                               // 000000001614: 84868104
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s6                                       // 00000000161C: 80020602
	s_addc_u32 s3, s3, s7                                      // 000000001620: 82030703
	global_load_u16 v1, v0, s[2:3]                             // 000000001624: DC4A0000 01020000
	s_waitcnt vmcnt(0)                                         // 00000000162C: BF8903F7
	v_cvt_f32_u32_e32 v1, v1                                   // 000000001630: 7E020D01
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001634: BF870091
	v_cmp_class_f32_e64 s2, v1, 0x90                           // 000000001638: D47E0002 0001FF01 00000090
	v_cndmask_b32_e64 v2, 1.0, 0x4f800000, s2                  // 000000001644: D5010002 0009FEF2 4F800000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001650: BF8704B1
	v_mul_f32_e32 v1, v2, v1                                   // 000000001654: 10020302
	v_cndmask_b32_e64 v2, 0, 0x42000000, s2                    // 000000001658: D5010002 0009FE80 42000000
	s_lshl_b64 s[2:3], s[4:5], 2                               // 000000001664: 84828204
	s_add_u32 s0, s0, s2                                       // 000000001668: 80000200
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 00000000166C: BF8700C2
	v_log_f32_e32 v1, v1                                       // 000000001670: 7E024F01
	s_addc_u32 s1, s1, s3                                      // 000000001674: 82010301
	s_waitcnt_depctr 0xfff                                     // 000000001678: BF880FFF
	v_sub_f32_e32 v1, v1, v2                                   // 00000000167C: 08020501
	v_mul_f32_e32 v1, 0x3f317218, v1                           // 000000001680: 100202FF 3F317218
	global_store_b32 v0, v1, s[0:1]                            // 000000001688: DC6A0000 00000100
	s_nop 0                                                    // 000000001690: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001694: BFB60003
	s_endpgm                                                   // 000000001698: BFB00000
