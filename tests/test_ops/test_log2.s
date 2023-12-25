
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_2925n20>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001610: BF870009
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001614: 84848204
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s4                                       // 00000000161C: 80020402
	s_addc_u32 s3, s3, s5                                      // 000000001620: 82030503
	s_add_u32 s0, s0, s4                                       // 000000001624: 80000400
	s_load_b32 s2, s[2:3], null                                // 000000001628: F4000081 F8000000
	s_addc_u32 s1, s1, s5                                      // 000000001630: 82010501
	s_waitcnt lgkmcnt(0)                                       // 000000001634: BF89FC07
	v_cmp_class_f32_e64 s3, s2, 0x90                           // 000000001638: D47E0003 0001FE02 00000090
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001644: BF870121
	v_cndmask_b32_e64 v0, 1.0, 0x4f800000, s3                  // 000000001648: D5010000 000DFEF2 4F800000
	v_cndmask_b32_e64 v1, 0, 0x42000000, s3                    // 000000001654: D5010001 000DFE80 42000000
	v_mul_f32_e32 v0, s2, v0                                   // 000000001660: 10000002
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001664: BF8700B1
	v_log_f32_e32 v0, v0                                       // 000000001668: 7E004F00
	s_waitcnt_depctr 0xfff                                     // 00000000166C: BF880FFF
	v_dual_sub_f32 v0, v0, v1 :: v_dual_mov_b32 v1, 0          // 000000001670: C9500300 00000080
	v_mul_f32_e32 v0, 0x3f317218, v0                           // 000000001678: 100000FF 3F317218
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001680: BF870001
	v_mul_f32_e32 v0, 0x3fb8aa3b, v0                           // 000000001684: 100000FF 3FB8AA3B
	global_store_b32 v1, v0, s[0:1]                            // 00000000168C: DC6A0000 00000001
	s_nop 0                                                    // 000000001694: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001698: BFB60003
	s_endpgm                                                   // 00000000169C: BFB00000
